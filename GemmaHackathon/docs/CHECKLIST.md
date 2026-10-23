# Checklists

## Night before (Oct 23)

### Software
- [ ] Run `CLAUDE_CODE_PROMPT.md` through Claude Code in this folder; `pytest -q` green
- [ ] Laptop: venv + `requirements-laptop.txt`
- [ ] Laptop: `scripts/download_models.sh` (E4B `.litertlm`)
- [ ] Laptop: `ollama pull gemma4:e4b` (fallback)
- [ ] Laptop: `python -m scripts.probe_litert` — record what works
- [ ] Laptop: `python -m scripts.test_gemma` — latency: ____ s
- [ ] Pi: 64-bit OS flashed, `bash pi/setup_pi.sh`
- [ ] Pi: `probe_litert --audio` — record what works
- [ ] Pi: `voice_agent.py --text` round trip — latency: ____ s
- [ ] Kortex wheel downloaded and installed
- [ ] Optional: Vosk small English model on the Pi

### Hardware
- [ ] Confirm arm: Gen3 / Gen3 lite
- [ ] `scripts/print_tags.py` → print tags 0–3, 10, 11 at 5 cm (+ spares)
- [ ] Webcam mount (clamp / tripod / boom), USB extension
- [ ] Pi 5 + 27 W PSU + cooler + SD card
- [ ] USB mic, small speaker, push button + jumper wires
- [ ] Ethernet cable ×2 (Pi, arm), USB-Ethernet adapter for laptop
- [ ] Power strip, tape, tape measure, marker
- [ ] 2 bins; 10–12 soft objects with obvious categories (fruit, tools, red/blue)
- [ ] Laptop charger

## Day of (Oct 24)

- [ ] 10:00 bench claimed, power, arm mounted and clamped
- [ ] Tags taped, `TABLE_TAGS` measured, `perception.calibrate` CALIBRATED
- [ ] 2:00 M1 click-to-pick
- [ ] 4:00 M2 Gemma-driven pick + Pi "go"
- [ ] 6:00 M3 teach + explain + correct
- [ ] Backup video recorded
- [ ] 7:00 project draft saved
- [ ] Rehearse ×3, assign who speaks, who holds the E-stop
