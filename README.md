# Gemma Hackathon

Two robot-arm projects built for the **Google Gemma Hardware Hackathon** (AGI House, Oct 24 2026), both running **Gemma fully on-device** against a **Kinova Gen3** arm — no cloud calls anywhere in the loop.

## Projects

### [GemmaHackathon](GemmaHackathon/) — "Show Me Once"
A robot arm that learns a sorting chore from a single human demonstration, explains the rule it inferred out loud, and then repeats it on new objects.

A swarm of on-device nodes on one Ethernet switch:
- **Raspberry Pi 5 (ears + mouth)** runs Gemma 4 E2B (LiteRT-LM) as a voice interface — audio in, intent JSON out.
- **Jetson Orin Nano Super (eyes)** reads the Kinova's wrist camera, locates objects and bins with AprilTags, and names each object with Gemma 4 E2B (Ollama, CUDA).
- **Laptop (RTX 4070, brain + hands)** runs Gemma 4 E4B (LiteRT-LM, with an Ollama fallback) as the planner and drives the arm.
- Gemma only chooses *which numbered object goes in which bin*; deterministic code turns that into a Kinova Kortex pick-and-place.
- Supports live correction: "actually, red things go in B" rewrites the rule on the fly.

See [GemmaHackathon/README.md](GemmaHackathon/README.md). It's one step-by-step guide covering setup, AprilTags, calibration, running, the demo, and troubleshooting.

### [GemmaHackathonBasic](GemmaHackathonBasic/) — Gemma ⇄ Kinova chat
A simpler chat interface to the same arm: type plain-English commands ("move the hand 5 cm forward", "rotate the wrist 20 degrees", "go home") and Gemma 4 (LiteRT-LM, laptop-only) turns them into validated, speed- and workspace-limited Kortex moves.

See [GemmaHackathonBasic/README.md](GemmaHackathonBasic/README.md) for setup and usage.

### [EdgeHand](EdgeHand/) — air-gapped tool handover
"Hand me the Phillips": Gemma 4 (LiteRT-LM) names the tool, a Jetson with an overhead camera finds it, and ROS 2 Humble + MoveIt 2 has the Kinova grasp the shaft and present the handle. The gripper opens when the operator tugs it, detected from the arm's joint torques. A Raspberry Pi provides request, release and stop buttons plus a link watchdog.

See [EdgeHand/README.md](EdgeHand/README.md) for the spec (v2, with what changed from v1), setup and the test results.

## Hardware (shared)

| Item | Role |
|---|---|
| Laptop, NVIDIA RTX 4070 (8 GB) | Runs Gemma (LiteRT-LM) + arm control |
| Kinova Gen3 / Gen3 lite | Manipulator (Kortex API), Ethernet-connected |
| Raspberry Pi 5 | Voice agent (GemmaHackathon only) |
| Jetson Orin Nano Super | Perception node: camera, AprilTags, object labels (GemmaHackathon only) |
| Gigabit Ethernet switch | Connects arm, laptop, Pi and Jetson (GemmaHackathon only) |
| Kinova Vision module (wrist camera) + AprilTags | Scene perception (GemmaHackathon only) |

## License

MIT for code written at the event. Gemma models are subject to the [Gemma terms of use](https://ai.google.dev/gemma/terms).
