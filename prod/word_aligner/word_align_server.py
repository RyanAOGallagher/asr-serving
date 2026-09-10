"""Streaming forced-alignment WS server — DROP-IN for the phoneme server protocol.
Per connection on WS /stream:
  1. client -> {"type":"text","text":"<transcript>","sampleRate":24000}   (NEW: text-first)
  2. client -> binary PCM16 mono audio chunks (at sampleRate)
  3. client -> {"type":"eos"}
Server echoes audio back (so the wrapper relays it to the client) and attaches
word timestamps as they settle, in the same envelope the wrapper already handles:
  -> {"type":"audio","audio":"<b64 echo>","words":[{"word","startS","endS"}]}
  -> {"type":"end","totalWords":N}
Online Viterbi forced alignment on the fine-tuned XLS-R char model. No torchaudio.
Run: CUDA_VISIBLE_DEVICES=1 uvicorn stream_align_server:app --host 0.0.0.0 --port 8009
"""
import os, re, json, base64, unicodedata
import numpy as np, torch
import torch.nn.functional as TF
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SR=16000; LOOKAHEAD=7; BEAM=15.0; NEG=-1e30
CKPT=os.environ.get("ALIGN_CKPT","/workspace/align_xlsr_best")
DEV=os.environ.get("ALIGN_DEVICE","cuda")
proc=Wav2Vec2Processor.from_pretrained(CKPT); TOK=proc.tokenizer
MODEL=Wav2Vec2ForCTC.from_pretrained(CKPT).to(DEV).eval(); BLANK=TOK.pad_token_id
MODEL(proc(np.zeros(SR,np.float32),sampling_rate=SR,return_tensors="pt").input_values.to(DEV))
app=FastAPI()

def normalize(t):
    t=unicodedata.normalize("NFC",t).lower(); t=re.sub(r"[^a-z0-9가-힣' ]"," ",t)
    return re.sub(r" +"," ",t).strip()
def to_jamo(t): return "".join(unicodedata.normalize("NFD",c) if "가"<=c<="힣" else c for c in t)
def from_jamo(t): return unicodedata.normalize("NFC",t)
def resample16k(x, sr):
    if sr==SR: return x
    n=int(round(len(x)*SR/sr))
    return TF.interpolate(torch.tensor(x,dtype=torch.float32).view(1,1,-1),size=n,mode="linear",align_corners=False).view(-1).numpy()

def viterbi(lp, states, target):
    T=lp.shape[0]; S=len(states); dp=np.full((T,S),NEG); bp=np.zeros((T,S),np.int32)
    dp[0,0]=lp[0,states[0]]
    if S>1: dp[0,1]=lp[0,states[1]]
    for t in range(1,T):
        pv=dp[t-1]
        for s in range(S):
            best,frm=pv[s],s
            if s-1>=0 and pv[s-1]>best: best,frm=pv[s-1],s-1
            if s-2>=0 and states[s]!=BLANK and states[s]!=states[s-2] and pv[s-2]>best: best,frm=pv[s-2],s-2
            dp[t,s]=best+lp[t,states[s]]; bp[t,s]=frm
    return dp,bp
def backtrace(dp,bp,f,anchor,states,n):
    s=anchor; te=[None]*n; ts=[None]*n
    for tt in range(f,-1,-1):
        if states[s]!=BLANK:
            ti=(s-1)//2
            if te[ti] is None: te[ti]=tt   # first seen going down = highest frame = END
            ts[ti]=tt                       # keep updating -> lowest frame = START
        s=int(bp[tt,s])
    return ts,te

class Aligner:
    def __init__(self, text):
        self.words=normalize(text).split(); self.target=[]; self.word_last=[]
        for w in self.words:
            for c in TOK(to_jamo(w)).input_ids: self.target.append(c)
            self.word_last.append(len(self.target)-1)
        self.states=[BLANK]
        for c in self.target: self.states+=[c,BLANK]
        self.buf=np.zeros(0,np.float32); self.committed=0; self.prev_end={}
        self.since=0  # samples added since last alignment pass (throttle)
    def add(self,x): self.buf=np.concatenate([self.buf,x]); self.since+=len(x)
    def _emit(self, ts, te, fs, frontier, final):
        out=[]
        n=len(self.words)
        while self.committed<n:
            i=self.word_last[self.committed]; le=te[i]
            stable = final or (le is not None and self.prev_end.get(i)==le and le<=frontier-LOOKAHEAD)
            if le is not None and stable:
                first_tok = (self.word_last[self.committed-1]+1) if self.committed>0 else 0
                beg = ts[first_tok]  # word's OWN acoustic start (after any pause), not prev word's end
                if beg is None: beg = (te[self.word_last[self.committed-1]] if self.committed>0 else 0) or 0
                out.append({"word":self.words[self.committed],"startS":round(beg*fs,3),"endS":round(le*fs,3)})
                self.committed+=1
            else: break
        return out
    def step(self, final=False):
        if not self.words: return []
        if not final and self.since < int(SR*0.35): return []   # throttle: re-align every ~0.35s, not every chunk
        self.since = 0
        if len(self.buf)<SR*0.12 and not final: return []
        if len(self.buf)<SR*0.04: return []
        x=proc(self.buf,sampling_rate=SR,return_tensors="pt").input_values.to(DEV)
        with torch.no_grad(): lp=torch.log_softmax(MODEL(x).logits,-1)[0].float().cpu().numpy()
        T=lp.shape[0]
        if T<2: return []
        fs=x.shape[1]/T/SR; dp,bp=viterbi(lp,self.states,self.target); S=len(self.states); f=T-1
        if final:
            anchor=S-1 if dp[f,S-1]>=dp[f,S-2] else S-2
        else:
            anchor=int(np.where(dp[f]>=dp[f].max()-BEAM)[0].max())
        ts,te=backtrace(dp,bp,f,anchor,self.states,len(self.target))
        out=self._emit(ts,te,fs,f,final)
        self.prev_end={i:e for i,e in enumerate(te) if e is not None}
        return out

@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept(); al=None; sr=24000; total=0
    try:
        while True:
            m=await ws.receive()
            if m["type"]=="websocket.disconnect": break
            if m.get("bytes"):
                chunk=m["bytes"]
                # echo audio FIRST (and flush) so playback never waits on alignment compute
                await ws.send_json({"type":"audio","audio":base64.b64encode(chunk).decode("ascii")})
                if al is not None:
                    pcm=np.frombuffer(chunk,dtype=np.int16).astype(np.float32)/32768.0
                    al.add(resample16k(pcm,sr)); w=al.step()
                    if w: total+=len(w); await ws.send_json({"type":"audio","audio":"","words":w})
            elif m.get("text"):
                msg=json.loads(m["text"])
                if msg.get("type")=="text":
                    sr=int(msg.get("sampleRate",24000)); al=Aligner(msg["text"]); total=0
                elif msg.get("type")=="eos":
                    if al is not None:
                        w=al.step(final=True)
                        if w: total+=len(w); await ws.send_json({"type":"audio","audio":"","words":w})
                    await ws.send_json({"type":"end","totalWords":total}); al=None
    except WebSocketDisconnect:
        pass
