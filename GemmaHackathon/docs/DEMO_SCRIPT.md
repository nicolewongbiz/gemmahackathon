# Demo day: timeline, script, pitch

## Event schedule (Oct 24, AGI House, 1868 Floribunda Ave)

| Time | Event | Our goal |
|---|---|---|
| 10:00 | Doors open | Claim a bench near power + space for the arm |
| 11:00 | Keynote | Pi + laptop booting, tags taped down |
| 12:00 | Pitches & team formation | (team already formed) |
| 12:15 | Building begins | Calibration first |
| 2:00 | — | **M1:** click-to-pick works |
| 4:00 | Check-ins | **M2:** frame → Gemma → arm picks (MVP) + Pi "go" |
| 6:00 | Dinner | **M3:** teach-from-demo + speech + correction |
| 7:00 | Save project draft | Draft saved, backup video recorded |
| 8:00 | Demos, judging, awards | Science-fair judging; top 6 on stage |

## Live demo (90 s)

1. **Hook (10 s):** "Robots today need a programmer. Ours needs one demonstration."
2. **Teach (20 s):** "Watch me." Sort 4 objects into A and B. "Done."
3. **Explain (10 s):** Pi speaks the rule. Point to the laptop screen showing the numbered marks.
4. **Do (30 s):** Invite a judge to dump new objects. "Go." Arm sorts, narrating.
5. **Correct (10 s):** "Actually, red things go in B." Pi confirms; arm moves one item.
6. **Offline proof (10 s):** Unplug Wi-Fi / airplane mode. "Two Gemma models, a Raspberry Pi and a laptop. Zero cloud."

## Pitch talking points

- **Problem:** teaching a robot a new task means code or hundreds of demos.
- **Insight:** a VLM can infer the *rule* from one demo if you give it the right interface — numbered marks, not coordinates.
- **On-device:** Gemma 4 E2B on LiteRT-LM on a Raspberry Pi 5 listens; Gemma 4 E4B on LiteRT-LM on a laptop GPU plans. Quote real latencies from `data/gemma_log.jsonl`.
- **Safe by design:** the model picks *what*, deterministic code decides *where* and *how*; bounds checks and an instant voice stop.
- **Where it goes:** home assistance, warehouse kitting, lab sample sorting, accessibility.

## Judge Q&A prep

| Question | Answer |
|---|---|
| Why not have Gemma output coordinates? | VLM pointing is imprecise; set-of-marks + homography is millimetre-accurate and safe. |
| What runs where? | Pi: E2B voice via LiteRT-LM CPU. Laptop: E4B vision via LiteRT-LM GPU. |
| Latency? | Read from the log: intent X s on Pi, plan Y s per step. |
| What if it's wrong? | Spoken correction updates the rule; invalid ids are rejected; stop is instant. |
| Generalises? | Rule is abstract ("food vs tools"), so new unseen objects sort correctly. |
| Why LiteRT? | Same runtime on a $80 board and a laptop GPU; small memory footprint (~1.4 GB for E2B). |

## Backup plan

- Record a clean 60-s video of a full run before 7:00.
- If the Pi fails: drive from the web UI at `http://<laptop>:8000` or `scripts/demo_cli.py`.
- If LiteRT vision fails on the laptop: `SMO_GEMMA_BACKEND=ollama`.
- If the arm faults: show `MockArm` + live frame + Gemma decisions, then the video.

## Submission checklist

- [ ] Project draft saved by 7:00 PM
- [ ] Main track: **Physical AI, IoT & Robotics**
- [ ] Mention LiteRT-LM use on both devices (LiteRT prize), Raspberry Pi role (Pi prize)
- [ ] Repo link, 60-s video, one architecture image
