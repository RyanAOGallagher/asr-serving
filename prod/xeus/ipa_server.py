"""Xeus + ZIPA IPA server (rebuilt after RunPod shutdown).

Same contract as the old RunPod ipa_server.py endpoints:
  POST /xeus  multipart field "files" (one or more audio files, any ffmpeg-readable format)
  -> [{"filename": ..., "ipa": ...}]
  POST /zipa  same contract. ZIPA-large (anyspeech/zipa-large-crctc-500k) via
  sherpa-onnx int8 on CPU — restored 2026-08-20 without touching VRAM (the
  reason it was originally dropped); ~0.1 s per clip on 4 threads.
  A 1 s silence tail is appended before decoding, matching how the June
  calibration runs (gen_zipa_ipa.py) fed the model.
  POST /wav2vec2  same contract. facebook/wav2vec2-lv-60-espeak-cv-ft via
  transformers (the path the June S2P bakeoff used), fp16 on GPU: 0.67 GB
  VRAM, ~0.2 s per clip. CPU is 20x slower, hence GPU.

Env: XEUS_DTYPE=float32|float16 (default float16), XEUS_MAX_SEC (default 60),
ZIPA_DIR (default: the cached anyspeech_zipa-large-crctc-500k snapshot),
ZIPA_THREADS (default 4), W2V_DTYPE (default float16).
GPU is shared with tm_stt workers — cache is freed after every request and
audio is truncated to XEUS_MAX_SEC so activations can't balloon the footprint
(a 44 s clip peaked at 3.2 GB in fp16).
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sherpa_onnx
import torch
from fastapi import FastAPI, File, UploadFile
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.model.xeusphoneme.builders import build_xeus_pr_inference

app = FastAPI()
inf = None
zipa_rec = None
w2v_proc = None
w2v_model = None


@app.on_event("startup")
def load_model():
    global inf, zipa_rec, w2v_proc, w2v_model
    ckpt = hf_hub_download("changelinglab/PhoneticXeus", "phoneticxeus_state_dict.pt")
    vocab = hf_hub_download("changelinglab/PhoneticXeus", "ipa_vocab.json")
    inf = build_xeus_pr_inference(
        work_dir=str(Path.home() / "ipa_asr" / "cache" / "xeus"),
        hf_repo="espnet/xeus",
        checkpoint=ckpt,
        vocab_file=vocab,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=os.environ.get("XEUS_DTYPE", "float16"),
        interctc_weight=0.3,
        interctc_layer_idx=[4, 8, 12],
        interctc_use_conditioning=True,
    )
    zipa_dir = os.environ.get(
        "ZIPA_DIR",
        str(Path.home() / "ipa_asr" / "cache" / "zipa" / "anyspeech_zipa-large-crctc-500k"),
    )
    zipa_rec = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=f"{zipa_dir}/model.int8.onnx",
        tokens=f"{zipa_dir}/tokens.txt",
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        num_threads=int(os.environ.get("ZIPA_THREADS", "4")),
    )

    # the 2.4 GB wav2vec2 snapshot lives in ipa_asr/hf_cache, not the default
    # HF cache the rest of this service uses — point at it explicitly rather
    # than setting HF_HOME (which would strand xeus's own cached weights)
    w2v_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
    w2v_cache = os.environ.get("W2V_CACHE", str(Path.home() / "ipa_asr" / "hf_cache" / "hub"))
    w2v_dtype = getattr(torch, os.environ.get("W2V_DTYPE", "float16"))
    w2v_proc = Wav2Vec2Processor.from_pretrained(w2v_name, cache_dir=w2v_cache)
    w2v_model = (
        Wav2Vec2ForCTC.from_pretrained(w2v_name, cache_dir=w2v_cache, dtype=w2v_dtype)
        .to("cuda" if torch.cuda.is_available() else "cpu")
        .eval()
    )


def decode_audio(data: bytes, suffix: str) -> np.ndarray:
    # m4a/mp4 needs a seekable input, so go through a temp file, not a pipe
    with tempfile.NamedTemporaryFile(suffix=suffix or ".bin") as tmp:
        tmp.write(data)
        tmp.flush()
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp.name,
             "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    if p.returncode != 0 or not p.stdout:
        raise ValueError(f"ffmpeg decode failed: {p.stderr.decode()[:200]}")
    wav = np.frombuffer(p.stdout, dtype=np.float32).copy()
    max_samples = int(float(os.environ.get("XEUS_MAX_SEC", "60")) * 16000)
    wav = wav[:max_samples]
    if len(wav) < 8000:
        wav = np.pad(wav, (0, 8000 - len(wav)))
    return wav


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "model_loaded": inf is not None,
        "zipa_loaded": zipa_rec is not None,
        "w2v_loaded": w2v_model is not None,
    }


@app.post("/wav2vec2")
async def wav2vec2(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        suffix = Path(f.filename or "audio.wav").suffix
        try:
            wav = decode_audio(await f.read(), suffix)
            with torch.no_grad():
                inputs = w2v_proc(wav, sampling_rate=16000, return_tensors="pt", padding=True)
                values = inputs.input_values.to(w2v_model.device).to(w2v_model.dtype)
                ids = torch.argmax(w2v_model(values).logits, dim=-1)
                ipa = w2v_proc.batch_decode(ids)[0]
            out.append({"filename": f.filename, "ipa": ipa.replace(" ", "")})
        except Exception as e:
            out.append({"filename": f.filename, "ipa": "", "error": str(e)[:300]})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


@app.post("/zipa")
async def zipa(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        suffix = Path(f.filename or "audio.wav").suffix
        try:
            wav = decode_audio(await f.read(), suffix)
            wav = np.concatenate([wav, np.zeros(16000, dtype=np.float32)])
            s = zipa_rec.create_stream()
            s.accept_waveform(16000, wav)
            zipa_rec.decode_stream(s)
            out.append({"filename": f.filename, "ipa": s.result.text.replace(" ", "")})
        except Exception as e:
            out.append({"filename": f.filename, "ipa": "", "error": str(e)[:300]})
    return out


@app.post("/xeus")
async def xeus(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        suffix = Path(f.filename or "audio.wav").suffix
        try:
            wav = decode_audio(await f.read(), suffix)
            res = inf(wav)
            out.append({"filename": f.filename, "ipa": res[0]["processed_transcript"]})
        except Exception as e:
            out.append({"filename": f.filename, "ipa": "", "error": str(e)[:300]})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
