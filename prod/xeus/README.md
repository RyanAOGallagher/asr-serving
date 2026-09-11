# Xeus IPA server (3060-server-1)

The phone recognizer the anchor calls as its **silence witness** (trial D, 2026-09-11) and as the
Korean-band tiebreak. Lives on `3060-server-1` (192.168.1.239, ssh alias in `~/.ssh/config`), not on
the ASR box. Anchor reaches it via `ANCHOR_XEUS_URL` (default `http://192.168.1.239:8001/xeus`).

| piece | where |
|---|---|
| service | `~/ipa_asr/ipa_server.py` (copied here verbatim) |
| launch | hand-run uvicorn, `PYTHONPATH=~/ipa_asr/PhoneticXeus XEUS_DTYPE=float16`, port 8001. `run_ipa_server.sh` here reproduces it; the server has no script, unit file, or cron. **Does not survive reboot.** |
| venv | `~/ipa_asr/venv`, Python 3.12.3, freeze in `requirements-3060-server-1.txt` (torch 2.12, transformers 5.11, sherpa-onnx 1.13.6, fastapi 0.136) |
| PhoneticXeus | `~/ipa_asr/PhoneticXeus`, upstream https://github.com/changelinglab/PhoneticXeus (paper arXiv 2603.29042). Plain copy, not a git checkout; 3.5 MB, 112 .py files, tree hash `sha256(sorted sha256s of src+configs) = 85ea9afc14d3d641`. Not vendored here. |
| weights | HF `changelinglab/PhoneticXeus` (state dict + `ipa_vocab.json`), `espnet/xeus` encoder, in the default HF cache; ZIPA `anyspeech_zipa-large-crctc-500k` int8 ONNX in `~/ipa_asr/cache/zipa`; wav2vec2 `facebook/wav2vec2-lv-60-espeak-cv-ft` in `~/ipa_asr/hf_cache`. ~12 GB total, not in git. |
| host | RTX 3060 12 GB (shared with tm_stt workers; service frees CUDA cache per request), driver 580, CUDA 12.0, ffmpeg 6.1 |

## Endpoints
`POST /xeus`, `POST /zipa`, `POST /wav2vec2`: multipart field `files` (one or more, any ffmpeg format)
-> `[{"filename", "ipa"}]`. `GET /healthz`. Audio is resampled to 16 kHz mono, truncated to `XEUS_MAX_SEC`
(60), zero-padded to at least 0.5 s. `/zipa` appends a 1 s silence tail before decoding.

## Behaviour the anchor relies on
- `/xeus` returns `""` on silence and on ~1 s one-syllable clips regardless of level (input is
  normalised internally; 100x gain changes nothing). The anchor treats `""` as no_speech when the EN
  decode confidence is below `ANCHOR_XEUS_SILENCE_CONF`. Any phones are never used to score text.
- Anchor failure mode: HTTP error or 10 s timeout -> `None` -> no evidence, text ships as before.
  If this box is down the silence rule silently stops working; nothing alerts.

## Restart
```
ssh 3060-server-1
cd ~/ipa_asr
kill $(ss -ltnp | grep ':8001 ' | grep -oP 'pid=\K[0-9]+')
(PYTHONPATH=$PWD/PhoneticXeus XEUS_DTYPE=float16 setsid nohup ./venv/bin/uvicorn ipa_server:app --host 0.0.0.0 --port 8001 >> ipa_server.log 2>&1 < /dev/null &)
curl -s localhost:8001/healthz
```
Model load takes ~45 s. `ipa_server.py.bak-20260820` on the box is the Xeus-only version before ZIPA/wav2vec2 were added back.

Bench scripts that lived beside it (`bench_dropin_*.py`, `bench_ttfp*.py`) are already in `work/paper/`.
