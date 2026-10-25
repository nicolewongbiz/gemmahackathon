# EdgeHand: Phase-Aware Robotic Instrument Handoff with On-Device Multimodal Gemma 4

*The scrub nurse that never looks away: it watches, predicts, and hands it over, fully offline.*

Gemma Hardware Hackathon, AGI House, Oct 24 2026. Track 2 (Gemma in IoT / Robotics / Embedded) and Best Use of LiteRT. Submission text for the project page is in [section 9](#9-submission-copy-project-page).

> **The idea in one line:** a robot arm that watches the surgery, works out which instrument comes next, *asks* the surgeon, and hands it over handle-first. Gemma 4 runs on-device through LiteRT-LM, and nothing leaves the room.

---

## 1. System at a glance

```
 ┌──────────────────────────── JETSON ORIN NANO SUPER (192.168.55.1) ─────────────────────────────┐
 │  surgical_feed.py   public aneurysm-clipping clip plays as the "operating microscope" camera     │
 │                     ├─ MJPEG video  :8090  ─────────────────────────────────────────┐           │
 │                     └─ phase events (phase timeline)  UDP :5007 ───────────────────┐     │      │   │
 │  cam_relay.py       Intel RealSense D435 on the arm's wrist → colour MJPEG :8089 ──┼─────┼──────┐   │
 │  network bridge     arm Ethernet → Jetson RJ45 (192.168.1.12, NAT) → USB-C → Mac  │     │      │   │
 └──────────────────────────────────────────────────────┬───────────────────────────┼─────┼──────┼───┘
                                                        │ USB-C (one cable)           │     │      │
 ┌──────────────────────────── MACBOOK (M-series)  192.168.55.100 ───────────────────▼─────▼──────┼───┐
 │  assistant.py   ── Gemma 4 E4B on LiteRT-LM (GPU) ─────────────────────────────────────────────│   │
 │     1. SCENE   (image)  frame → instruments in view, aneurysm visible?, clip applied?, phase read │   │
 │     2. DECIDE  (text)   scene + phase + tray → instrument + reason                               │   │
 │     3. SPEAK   macOS `say` (offline): "Clipping. Shall I bring the green clip applier?"          │   │
 │     4. LISTEN  mic → Gemma audio → transcript → yes / no / "the suction" / stop / let go          │   │
 │        └─ on "yes" → request  UDP :5006 ──┐          ▲ arm_state (busy/holding/idle) UDP :5007   │   │
 │  run_kortex.py  (Kortex API, no ROS)  ◄────┘──────────┘                                           │   │
 │     look pose → wrist camera (via Jetson :8089) → AprilTags → table map → find + name tool  ◄─────┘   │
 │     → safety checks → pick by the handle → present → release on the surgeon's tug                    │
 └──────────────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                        │ Kortex API over Ethernet (routed through the Jetson)
                                               ┌────────▼────────┐
                                               │ KINOVA GEN3 7-DoF│  Robotiq 2F-85 gripper, RealSense on the wrist,
                                               │  192.168.1.10   │  red E-stop in a human hand
                                               └─────────────────┘
```

## 2. Where everything runs

| Machine | Runs | Talks to |
|---|---|---|
| **MacBook** (Apple M-series) | `assistant.py`: **all Gemma inference** (LiteRT-LM, GPU), the stage screen, mic + speech. `run_kortex.py`: the arm, wrist-camera vision, every safety check | Jetson over one USB-C cable; the arm through the Jetson |
| **Jetson Orin Nano Super** (JetPack 6.2) | `surgical_feed.py`: the surgical clip as a live camera feed + phase events. `cam_relay.py`: the RealSense colour stream. Network bridge to the arm | Mac (USB-C, 192.168.55.x); arm (Ethernet, 192.168.1.x) |
| **Kinova Gen3** + Robotiq 2F-85 | Motion through the Kortex API (Cartesian reaches, gripper, joint torques) | Mac, via the Jetson |
| **Intel RealSense D435** (on the wrist) | Colour camera for finding the instruments on the tray | Jetson (USB 3) |

**Ports.** UDP 5006 carries requests to the arm runner. UDP 5007 carries phase events and arm state to the assistant. HTTP 8089 is the wrist-camera stream and HTTP 8090 is the surgical-video stream. There's no internet and no cloud anywhere.

**Why Gemma runs on the Mac and not the Jetson.** We tested Gemma 4 on the Orin Nano through LiteRT. The GPU path ran out of memory (Vulkan OOM on the 8 GB board) and the CPU path took 31 s per text request. On the Mac's GPU it takes ~1.5 s for text and ~6.5 s with an image. So the Jetson does what it's best at, video and I/O, and the Mac does the reasoning.

## 3. End-to-end flow (one handover)

| # | Step | Where | Time |
|---|---|---|---|
| 1 | The clip reaches a new phase, e.g. *Clipping*. The Jetson sends `{"type": "phase", "phase": "Clipping"}` | Jetson | — |
| 2 | **Gemma, scene:** looks at the current microscope frame *without being told the phase*: "3 instruments: forceps, clip applier, micro-dissector; aneurysm visible: no; clip applied: yes; my read: Clipping" | Mac, Gemma (vision) | ~6.5 s |
| 3 | **Gemma, decide:** scene report + phase + tray → "green clip applier: clipping phase requires the clip applier to place or adjust clips" | Mac, Gemma (text) | ~1.7 s |
| 4 | **Ask:** the laptop says *"Clipping. Shall I bring the green clip applier?"* If the arm is busy, it waits and asks when the arm is free | Mac, `say` | ~2 s |
| 5 | **Listen 3.5 s:** Gemma transcribes; fixed keywords decide yes / no / stop / let go; Gemma maps a named alternative ("no, the suction") | Mac, Gemma (audio) | ~1–3 s |
| 6 | On **yes**: request → arm runner. The arm moves to the look pose; the wrist camera finds the four instruments (AprilTag table map, colour accent naming) | Mac + Jetson camera | a few s |
| 7 | **Pick:** rise, travel high, descend, slow last 3 cm, close on the handle, lift, carry. Every target is checked against the measured table plane | Arm | 20–40 s |
| 8 | **Present** at chest height, handle toward the surgeon. **Release on a tug** (wrist torque change) or "let go" | Arm | until taken |
| — | **Voice any time** (press `v`): "I need suction", "stop", "let go" | Mac | ~1–3 s |

## 4. Where Gemma comes in

All of it is **Gemma 4 E4B on LiteRT-LM, on the laptop**. One model handles vision, audio and text.

| Role | Input → output | Guardrail |
|---|---|---|
| **Surgical scene understanding** | microscope frame → instruments in view, aneurysm visible, clip applied, its own phase read | It isn't told the phase, so its read is independent; it's shown next to the timeline's phase |
| **Instrument decision** | scene + phase + tray → instrument + one-line reason | It picks from 4 colours only (whitelisted); it never outputs coordinates; a human must say yes |
| **Listening** | microphone audio → transcript (neutral prompt) | Stop, release, yes and no come from fixed keywords, never from Gemma's judgement |
| **Request mapping** | "no, give me something to coagulate" → yellow bipolar forceps | Whitelisted to the tray |
| **Typed requests and tool naming** (fallback) | text → tool class; tray image → tool names | Used when the assistant isn't running. During the demo the arm names tools by colour, so only one process uses the GPU |

**Gemma chooses *what*; deterministic code decides *how*.** Geometry comes from the AprilTags and the measured table plane. Motion goes through the Kortex API with clearance checks. Gemma can't move the arm on its own.

## 5. Safety, in layers

1. **The red E-stop** is in a human hand at all times.
2. **Human in the loop:** Gemma's suggestion moves nothing until the surgeon says "yes".
3. **Safety words without the model:** "stop", "wait", "let go" are matched by fixed keywords on the transcript. (In testing, a combined prompt made Gemma hear "stop" as "clip applier". So we split hearing from deciding.)
4. **The table is never touched:** the table plane is measured with fingertip touches on the four tags. Any grasp that would put the closed fingertips less than 4 mm above it is refused. The last 3 cm run at 6% speed.
5. **Motion style:** reaches are slow (20% speed) and rise before moving sideways, so nothing travels low over the tray.
6. **Release only on intent:** the gripper opens on a deliberate tug (joint-torque change, ~100× the sensor noise) or a spoken "let go".

## 6. What's live, and what isn't

| | Status |
|---|---|
| Gemma's scene analysis, instrument choice, speech, listening | **Live**, on-device |
| Wrist-camera vision, AprilTag calibration, the arm, the tug release | **Live**, real hardware |
| **The phase timeline** | Prepared for a public, licensed clip. Our phase-recognition model is real (see below), but its weights live with protected clinical data, so we can't show it running |
| The surgical video | A public clip played by the Jetson in place of a real microscope feed |
| The instruments | Colour-coded screwdrivers standing in for surgical tools |

## 7. Results

| What | Result |
|---|---|
| **Neurosurgical phase model** (offline, `NeurosurgicalPhaseRecognition/`): 48 expert-annotated aneurysm-clipping videos, 40,725 frames, held-out-video split | **95.7%** frame accuracy (ResNet50 + MS-TCN + Viterbi) |
| Gemma's **zero-shot** phase read on the demo clip | 3/7 agree with the timeline, which is why a specialised model supplies the phase |
| Gemma request understanding (59 phrases, 6 tool classes) | 93% |
| Gemma naming tools in a tray image (synthetic, including tools not on the list) | 88% |
| Multi-tool sim end to end (MuJoCo, wrist camera + Gemma naming) | 29/30 |
| Voice: 6 test phrases; confirmation (yes / no / redirect / stop / unclear) | 6/6; 5/5 |
| Latency: scene + decision / voice command | ~8 s / 1–3 s |
| **Real arm:** full handovers end to end, including release on tug | **[fill in: N of M on the day]** |

---

## 8. The pitch (90 seconds)

**Hook, 10 s.**
"In the operating room, a scrub nurse has the next instrument ready before the surgeon asks. There aren't enough of them, and the OR is the one place you can't send video to the cloud. So we built a scrub nurse that runs entirely on the edge."

**What it does, 25 s, demo running.**
"The Jetson is our operating-microscope feed: a real aneurysm-clipping video. When the surgery moves into clipping, Gemma 4, running on this laptop through LiteRT, looks at the frame. It sees the instruments already in use and whether the aneurysm is exposed, and decides the surgeon needs the clip applier. It asks."
*(Laptop: "Clipping. Shall I bring the green clip applier?" You: "Yes.")*
"The arm finds it on the tray with its wrist camera, picks it by the handle, and holds it out. When I pull, it lets go."

**Why it's trustworthy, 20 s.**
"Gemma decides *what*, never *how*. It picks from a fixed list and never outputs a coordinate, and nothing moves without a spoken yes. Stop and let-go are hard-wired keywords, so they don't depend on the model. Geometry comes from calibrated tags, and the arm won't come within 4 mm of the table. There's a red E-stop in my hand."

**Why it's Gemma on the edge, 15 s.**
"Every model call, whether vision, speech or reasoning, is Gemma 4 on LiteRT-LM on this laptop. The Jetson handles the video. Pull the network cable and nothing changes." *(pull it)*

**Credibility, 15 s.**
"The phase signal comes from our own neurosurgical phase model: 95.7% on 48 expert-annotated surgeries. Its weights sit with protected clinical data, so today the phase timeline is prepared in advance. Gemma zero-shot only gets the phase 3 times out of 7. That's why we pair a specialised model for the phase with Gemma for the reasoning."

**Close, 5 s.**
"An offline scrub nurse: it watches, asks, and hands it over."

### Slides (4)
1. **The problem**: scrub-nurse shortage; OR data can't leave the room.
2. **The system diagram** (section 1), with Gemma's three roles highlighted: see, decide, listen.
3. **Safety layers** (section 5) + **what's live** (section 6).
4. **Results** (section 7) + "Gemma 4 on LiteRT-LM, fully offline".

### Judges' Q&A
- **Is the phase recognition live?** The phase model is real (95.7% on 48 surgeries, code in the repo), but its weights are tied to protected clinical data, so the demo replays a prepared timeline on a public clip. Everything Gemma does on top of it is live.
- **Why not have Gemma predict the phase?** We measured it: 3/7 zero-shot. A small specialised temporal model is far better at phases. Gemma is better at reasoning over the scene and talking to the surgeon.
- **What if Gemma is wrong?** It only suggests. The surgeon says no or names another instrument. Its outputs are restricted to the tray, and it never controls motion directly.
- **Is this a VLA?** No. That's deliberate: models choose what, deterministic code decides how. We think that's the right split near people.
- **Why not run Gemma on the Jetson?** We tried: LiteRT's GPU path ran out of memory on the 8 GB Orin Nano and the CPU path took 31 s per request. The laptop does it in 1.5–8 s, so the Jetson handles video and the laptop handles reasoning.
- **How fast is it?** About 8 s from a phase change to the question, 1–3 s for a voice command, and 20–40 s for a handover at a deliberately slow 20% speed.
- **What about patient data?** None is used or shown. The clip is public and licensed, and the clinical labels and weights never left their source.
- **What would you build next?** Run the phase model on the Jetson (TensorRT), feed Gemma's scene features into it alongside the CNN features, and use real surgical instruments with a 3D grasp from the RealSense depth stream.

---

## 9. Submission copy (project page)

### Title and tagline
> **EdgeHand: Phase-Aware Robotic Instrument Handoff with On-Device Multimodal Gemma 4**
> *The scrub nurse that never looks away: it watches, asks, and hands it over, fully offline.*

Other titles:
- EdgeHand: An Air-Gapped Multimodal Scrub-Nurse Robot for Neurosurgery (stresses offline and neurosurgery)
- EdgeHand: Phase-Conditioned, Human-Confirmed Instrument Delivery on the Edge (stresses the ask-first safety design)
- EdgeHand: Anticipatory Surgical Instrument Handoff with Gemma 4 on LiteRT (stresses the LiteRT prize)

Other taglines:
- *Watches. Asks. Hands it over. Nothing leaves the room.*
- *It anticipates the next instrument, and asks before it moves.*
- *Gemma sees the surgery, the surgeon says yes, and the arm delivers, fully on-device.*

### Describe your project

**EdgeHand: the scrub nurse that never looks away.** An offline robotic scrub nurse for neurosurgery, powered by Gemma 4 running entirely on the edge.

**Why neurosurgery needs this.** In an aneurysm clipping, the surgeon works through a microscope on blood vessels a few millimetres wide, right beside a bulge in the artery wall that can rupture. The critical moments come down to seconds and steady hands. Every time the surgeon looks away from the eyepiece, or waits for the right instrument, focus breaks at exactly the wrong time. A great scrub nurse prevents that by having the next instrument ready before it's asked for. That skill takes years to build, and experienced scrub nurses are scarce. The operating room is also the one place you can't send video to the cloud: patient privacy, unreliable connectivity, and zero tolerance for an outage mid-procedure. So the intelligence has to live in the room.

**What EdgeHand does.** It watches the surgery, recognises the phase, sees what's already in the field, anticipates the next instrument, and **asks**: *"Clipping. Shall I bring the clip applier?"* On a spoken "yes", a Kinova Gen3 arm finds the instrument on the tray with its wrist camera, picks it by the handle, and presents it handle-first. When the surgeon pulls, it lets go. The surgeon can speak to it at any time: "no, the suction", "stop", "let go".

**Where Gemma comes in.** One Gemma 4 model handles vision, audio and reasoning, on-device through LiteRT-LM:
- **It sees:** it reads the microscope frame, counts the instruments in use, and judges whether the aneurysm is exposed and whether a clip is already on.
- **It reasons:** it combines the surgical phase, the scene and the tray into a choice and a one-line reason.
- **It listens:** it understands the surgeon's voice directly from audio.

Pull the network cable and nothing changes.

**Built to be trusted near a patient.** Gemma decides *what*, never *how*:
- It chooses only from the instruments on the tray and never outputs a coordinate.
- Nothing moves without a spoken "yes".
- "Stop" and "let go" are hard-wired keywords that never depend on the model.
- The arm calibrates the table with fingertip touches and refuses any grasp that would bring it within 4 mm of the surface.
- A human holds the red E-stop throughout.

**Grounded in real neurosurgical AI.** The phase signal comes from our own surgical phase-recognition model: **95.7% accuracy on 48 expert-annotated aneurysm-clipping surgeries** (ResNet50 + MS-TCN). Its weights are tied to protected clinical data, so the demo replays a prepared phase timeline on a public surgical video. Everything Gemma does on top of it is live. We also measured Gemma reading phases on its own: 3 of 7 correct. That's exactly why EdgeHand pairs a specialised phase model with Gemma's reasoning and conversation.

**Next:** real surgical instruments with 3D grasping from the depth camera, the phase model running live on the Jetson, and testing with surgical teams.

### Tech stack

- **AI:** Gemma 4 E4B (vision, audio and text in one model), on-device via **Google LiteRT-LM** (GPU, fully offline)
- **Surgical phase recognition:** PyTorch, ResNet50 features + MS-TCN temporal model + Viterbi decoding (95.7% on 48 aneurysm-clipping videos)
- **Robot:** Kinova Gen3 7-DoF arm + Robotiq 2F-85 gripper, controlled through the **Kinova Kortex API** (Python)
- **Vision:** Intel RealSense D435 on the wrist, OpenCV, AprilTag table calibration, contrast segmentation, and colour-accent instrument identification
- **Edge compute:**
  - **NVIDIA Jetson Orin Nano Super:** surgical video node, camera relay, network bridge to the arm
  - **MacBook (Apple silicon):** Gemma inference, motion and safety
- **Voice:** Gemma native audio understanding, macOS on-device speech synthesis, keyword-gated safety commands
- **Networking:** air-gapped USB-C and Ethernet links, a lightweight UDP/JSON event bus, MJPEG video streams
- **Safety and testing:** joint-torque tug detection, fingertip-calibrated table plane, MuJoCo physics simulation of the full handover, and pytest unit and integration tests
- **Languages:** Python, with ROS 2 / MoveIt 2 designs for future clinical integration
