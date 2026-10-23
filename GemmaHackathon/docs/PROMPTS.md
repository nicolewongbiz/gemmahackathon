# Prompt design

## Set-of-marks

VLMs are poor at outputting precise pixel coordinates but good at choosing among labelled options. So the laptop:
1. finds object blobs with classical CV,
2. draws a large white circle with a black number on each,
3. labels bins "BIN A" / "BIN B",
4. asks Gemma to answer with a **number** and a **letter**.

Mark rules: circle radius ≥ 16 px at 768 px image width, black 2 px outline, numbers 1–N in reading order, never overlapping an object's own label/text.

## Prompts (`planner/prompts.py`)

**SYSTEM**
> You are the planner for a tabletop robot arm. Images show a table from above. Objects are marked with numbered white circles. Bins are labelled A and B. Always answer with a single JSON object and nothing else.

**TEACH** — N ordered keyframes of a person sorting → `{"rule", "bins": {"A","B"}, "say"}`. Ask for the *rule* (category, colour, material, size, use), not a list of items, so it generalises to new objects.

**PLAN** — rule + bin descriptions + the list of valid ids → one `{"object", "label", "bin", "reason"}`. One object per call keeps answers short and lets the scene update after every pick.

**CORRECT** — current rule + user's words → updated rule in TEACH shape.

**Pi INTENT (E2B)** — maps speech to one of `empty, teach, done, go, stop, correct, chat`, returns the user's words and a ≤ 6-word spoken reply.

## Reliability tricks

- Temperature 0.2.
- One short few-shot example per prompt.
- Extract the first `{...}`; on failure, retry with "Reply with ONLY a valid JSON object." (max 2).
- Whitelist: reject `object` not in the drawn ids; reject `bin` not in {A, B}.
- Keyframes ≤ 6, longest side ≤ 768 px to bound latency and VRAM.
- Log every call to `data/gemma_log.jsonl` (prompt hash, latency, raw output) — use it to tune prompts and to quote latency in the pitch.

## Ideas if time allows

- Ask Gemma for a confidence; if low, ask the user out loud ("Is the sponge a tool or food?").
- Use Gemma 4 tool calling (`tools=[pick, say]`) on LiteRT-LM instead of raw JSON.
- Let the Pi narrate with a slightly longer, friendlier reply when idle.
