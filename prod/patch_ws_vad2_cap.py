#!/usr/bin/env python3
"""Add a max-utterance safety cap to the VAD endpointing path (bounds re-decode cost
on no-pause speech). ?max_utterance_ms=0 disables the cap."""
import time, shutil

P = "server_merged.py"
s = open(P, encoding="utf-8").read()

# A) parse max_utterance_ms next to silence_ms
a_old = '''    try:
        silence_ms = int(ws.query_params.get("silence_ms") or 700)
    except (TypeError, ValueError):
        silence_ms = 700
'''
a_new = a_old + '''    try:
        max_utterance_ms = int(ws.query_params.get("max_utterance_ms") or 20000)
    except (TypeError, ValueError):
        max_utterance_ms = 20000
'''
assert s.count(a_old) == 1, f"A count={s.count(a_old)}"
s = s.replace(a_old, a_new)

# B) derive sample cap next to VAD_WIN
b_old = "    VAD_WIN = 512  # silero requires 512-sample windows at 16k\n"
b_new = b_old + "    max_utt_samples = int(max_utterance_ms / 1000 * SAMPLE_RATE) if max_utterance_ms > 0 else 0\n"
assert s.count(b_old) == 1, f"B count={s.count(b_old)}"
s = s.replace(b_old, b_new)

# C) force endpoint when the current utterance's audio exceeds the cap
c_old = '''                            elif "end" in ev and speech_seen:
                                endpointed = True
                                break
                    if endpointed:
'''
c_new = '''                            elif "end" in ev and speech_seen:
                                endpointed = True
                                break
                    if (not endpointed and max_utt_samples
                            and state.audio_accum.shape[0] >= max_utt_samples):
                        endpointed = True  # safety cap: bound re-decode cost on no-pause speech
                    if endpointed:
'''
assert s.count(c_old) == 1, f"C count={s.count(c_old)}"
s = s.replace(c_old, c_new)

bak = f"{P}.bak.vadcap.{int(time.time())}"
shutil.copyfile(P, bak)
open(P, "w", encoding="utf-8").write(s)
print("cap patch applied. backup:", bak)
