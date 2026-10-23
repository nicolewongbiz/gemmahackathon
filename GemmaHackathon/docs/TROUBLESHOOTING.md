# Troubleshooting

## LiteRT-LM

| Symptom | Fix |
|---|---|
| `TypeError` on `Engine(...)` kwargs | Run `python -m scripts.probe_litert`; update the construction list in the adapter. |
| Image/audio ignored or error | Content format differs from the guess; probe prints accepted helpers (`Contents.of`, etc.). |
| GPU backend fails on laptop | Try CPU backend for E4B (slower), or `SMO_GEMMA_BACKEND=ollama`. |
| `pip install litert-lm-api` fails on Pi | Must be 64-bit OS (`uname -m` → `aarch64`); try `litert-lm-api-nightly`. |
| Model download slow / 429 | Set `HF_TOKEN`; download the night before. |
| Pi OOM | Close the desktop session (`sudo systemctl set-default multi-user.target`), use E2B not E4B. |

## Gemma output

| Symptom | Fix |
|---|---|
| Non-JSON output | Retry is automatic; shorten the prompt; add/fix the few-shot example. |
| Picks an id that isn't drawn | Rejected by whitelist; make marks bigger, reduce image clutter. |
| Wrong rule from teach | Use objects with an obvious split; ensure each keyframe shows a new object in a bin; try 4 instead of 6 frames. |
| Slow (> 5 s/step) | `PROMPT_IMAGE_MAX=512`, fewer keyframes, check GPU is actually used (`nvidia-smi`). |

## Perception

| Symptom | Fix |
|---|---|
| NOT CALIBRATED | All of tags 0–3 must be visible; improve lighting; check printed size. |
| Arm occludes tags | Homography is cached; return arm home before each detection. |
| Shadows detected as objects | Raise `DIFF_THRESHOLD`; diffuse lighting; retake `empty`. |
| Two touching objects = one blob | Spread objects; raise `MIN_AREA_PX` doesn't help — use spacing. |
| Pick is offset | Re-measure `TABLE_TAGS`; tall objects shift — lower camera tilt; add per-axis offset in config. |

## Kinova

| Symptom | Fix |
|---|---|
| Can't connect | Static IP 192.168.1.11/24 on the right interface; ping 192.168.1.10; open the Web App. |
| `kortex_api` import error | Install the wheel; pin `protobuf==3.20.*`. |
| Action aborts | Target unreachable or in singularity → move bins/table closer; check `TOOL_ORIENT`. |
| Arm in fault | Clear faults in the Web App; `base.ClearFaults()`. |
| Gripper doesn't close | Gen3 lite vs Robotiq finger values differ; test 0.0–1.0 in `test_kinova.py`. |

## Network / Pi

| Symptom | Fix |
|---|---|
| Pi can't reach laptop | `ping 192.168.50.1`; laptop firewall inbound 8000; server running with `--host 0.0.0.0`. |
| No sound | `aplay -l`, set default card in `~/.asoundrc`, `espeak-ng "test"`. |
| Mic silent | `arecord -l`; pass `--mic <index>`; check USB power. |
