"""Step 1 sanity check: does Gemma see the webcam and return JSON?
    python -m scripts.test_gemma [litert|ollama]
"""
import sys, time
import config
from perception.camera import Camera
from planner import gemma_vision

backend = sys.argv[1] if len(sys.argv) > 1 else config.GEMMA_BACKEND
f = Camera().read()
t = time.time()
print(gemma_vision.ask_json('List the objects on the table. JSON: {"objects": ["..."]}', [f], backend=backend))
print(f"{backend}: {time.time() - t:.1f}s")
