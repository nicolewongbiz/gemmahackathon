# Gemma Hackathon

Built for the **Google Gemma Hardware Hackathon** (AGI House, Oct 24 2026).

## EdgeHand: air-gapped edge-AI tool-handover cell

A Kinova Gen3 hands an operator the tool they ask for, such as "hand me the Phillips screwdriver". The cell runs 100% offline: Gemma 4 on LiteRT-LM parses the request and names the tools it sees through an Intel RealSense on the arm's wrist, and the arm grasps the tool and presents the handle, releasing it when the operator tugs. No cloud calls anywhere in the loop.

**See [EdgeHand/README.md](EdgeHand/README.md) for everything: spec, setup, running, tests and troubleshooting.**

## Repo layout

| Path | What it is |
|---|---|
| [`EdgeHand/`](EdgeHand/) | The project. ROS 2 workspace, no-ROS Kortex runner, MuJoCo sim, tests. |
| [`GemmaHackathon/`](GemmaHackathon/) | The earlier prototype ("Show Me Once", a sorting demo). Kept because EdgeHand still uses parts of it: the MuJoCo sim assets and sim venv (`sim/menagerie`, `sim/.venv`), the Gemma model (`data/gemma-4-E4B-it.litertlm`), the laptop venv (`.venv`), the Kortex wheel and `requirements-laptop.txt`. Its own [README](GemmaHackathon/README.md) describes that prototype. |

## License

MIT for code written at the event. Gemma models are subject to the [Gemma terms of use](https://ai.google.dev/gemma/terms).
