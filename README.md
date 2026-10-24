# Gemma Hackathon

Two robot-arm projects built for the **Google Gemma Hardware Hackathon** (AGI House, Oct 24 2026), both running **Gemma fully on-device** against a **Kinova Gen3** arm — no cloud calls anywhere in the loop.

## Projects

### [GemmaHackathon](GemmaHackathon/) — "Show Me Once"
A robot arm that learns a sorting chore from a single human demonstration, explains the rule it inferred out loud, and then repeats it on new objects.

- **Raspberry Pi 5** runs Gemma 4 E2B (LiteRT-LM) as a voice interface — audio in, intent JSON out.
- **Laptop (RTX 4070)** runs Gemma 4 E4B (LiteRT-LM, with an Ollama fallback) as the vision planner.
- A USB webcam above the table + AprilTags locate objects and bins; Gemma only chooses *which numbered object goes in which bin*, deterministic code turns that into a Kinova Kortex pick-and-place.
- Supports live correction: "actually, red things go in B" rewrites the rule on the fly.

See [GemmaHackathon/README.md](GemmaHackathon/README.md). It's one step-by-step guide covering setup, AprilTags, calibration, running, the demo, and troubleshooting.

### [GemmaHackathonBasic](GemmaHackathonBasic/) — Gemma ⇄ Kinova chat
A simpler chat interface to the same arm: type plain-English commands ("move the hand 5 cm forward", "rotate the wrist 20 degrees", "go home") and Gemma 4 (LiteRT-LM, laptop-only) turns them into validated, speed- and workspace-limited Kortex moves.

See [GemmaHackathonBasic/README.md](GemmaHackathonBasic/README.md) for setup and usage.

## Hardware (shared)

| Item | Role |
|---|---|
| Laptop, NVIDIA RTX 4070 (8 GB) | Runs Gemma (LiteRT-LM) + arm control |
| Kinova Gen3 / Gen3 lite | Manipulator (Kortex API), Ethernet-connected |
| Raspberry Pi 5 | Voice agent (GemmaHackathon only) |
| USB webcam + AprilTags | Scene perception (GemmaHackathon only) |

## License

MIT for code written at the event. Gemma models are subject to the [Gemma terms of use](https://ai.google.dev/gemma/terms).
