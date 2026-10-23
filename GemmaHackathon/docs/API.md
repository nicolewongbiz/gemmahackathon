# API and JSON contracts

Base URL: `http://<laptop>:8000`

## Endpoints

### `POST /intent`
```json
{"intent": "empty | teach | done | go | stop | correct", "text": "optional user words"}
```
Response (returns immediately; long work runs in a background thread):
```json
{"ok": true, "mode": "idle", "rule": {"rule": "...", "bins": {"A": "...", "B": "..."}}}
```

| Intent | Allowed in | Effect |
|---|---|---|
| `empty` | idle | Snapshot the empty table (background for object detection) |
| `teach` | idle | Start recording keyframes |
| `done` | teaching | Send keyframes to Gemma → new rule → speak it |
| `go` | idle | Start the sort loop |
| `stop` | any | Halt arm immediately, return to idle |
| `correct` | idle | Gemma rewrites the rule from `text` |

### `GET /say`
Drains the speech queue. The Pi polls every 0.5 s.
```json
{"say": ["Watching. Show me.", "Food goes in A, tools go in B."]}
```

### `GET /status`
```json
{"mode": "sorting", "rule": {...}, "calibrated": true, "objects": 4,
 "last_step": {"object": 3, "bin": "A", "label": "banana"}, "last_latency_s": 1.8}
```

### `GET /frame.jpg`
Latest annotated frame (numbered marks, bins, chosen object highlighted).

### `GET /`
Backup web UI: live frame, status, buttons for each intent.

### `GET /docs`
Auto-generated FastAPI docs.

## Gemma JSON contracts

**Pi intent (E2B)**
```json
{"intent": "go", "text": "okay sort these", "reply": "On it."}
```
`intent ∈ {empty, teach, done, go, stop, correct, chat}`; `chat` is not forwarded.

**Teach (E4B)**
```json
{"rule": "Sort by category: food vs tools.", "bins": {"A": "food", "B": "tools"}, "say": "Food goes in A, tools go in B."}
```

**Plan step (E4B)**
```json
{"object": 3, "label": "banana", "bin": "A", "reason": "food goes in A"}
```
or, when nothing fits:
```json
{"object": null, "say": "The remaining items don't match either bin."}
```
`object` must be one of the ids drawn on the image; anything else is rejected.

**Correct (E4B)**
```json
{"rule": "Red things go in B, everything else by category.", "bins": {"A": "food (not red)", "B": "tools and red things"}, "say": "Got it, red things go in B."}
```

## curl examples

```bash
curl -X POST localhost:8000/intent -H 'content-type: application/json' -d '{"intent":"empty"}'
curl -X POST localhost:8000/intent -H 'content-type: application/json' -d '{"intent":"teach"}'
curl -X POST localhost:8000/intent -H 'content-type: application/json' -d '{"intent":"done"}'
curl -X POST localhost:8000/intent -H 'content-type: application/json' -d '{"intent":"go"}'
curl -X POST localhost:8000/intent -H 'content-type: application/json' -d '{"intent":"correct","text":"red things go in B"}'
curl -X POST localhost:8000/intent -H 'content-type: application/json' -d '{"intent":"stop"}'
```
