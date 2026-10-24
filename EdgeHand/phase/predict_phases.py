"""Run the neurosurgical phase model (github.com/veronicawang1/neurosurgical_phase_recognition) on one video and
write the phases.csv that edgehand/phase_player.py replays. Offline, before the demo.

Same pipeline as the repo: frames at 1 FPS -> fine-tuned ResNet50 features (2048-d) -> MS-TCN (2 stages, 8 layers,
128 filters, as in logs/mstcn_finetuned_v1) -> softmax -> Viterbi with the surgical ordering prior.

    python predict_phases.py --repo ../../neurosurgical_phase_recognition --video ~/Desktop/clippingdemo.mp4 \
        --backbone checkpoints/finetuned_backbone.pt --mstcn checkpoints/best_mstcn.pt --out ../data/phases/clippingdemo.csv

Output: time_s,phase,p_<phase>... one row per second. The learned transition matrix needs the training labels, which
aren't here, so Viterbi uses the repo's default ordering prior (stay 0.95, advance 0.04, skip 0.009, back 0.001).
"""
import argparse, csv, sys
from pathlib import Path
import cv2
import numpy as np
import torch

PHASES = ["Brain Exposure", "Parent Vessel Identification", "Dome and Neck Identification", "Clipping"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the neurosurgical_phase_recognition checkout")
    ap.add_argument("--video", required=True)
    ap.add_argument("--backbone", required=True, help="checkpoints/finetuned_backbone.pt")
    ap.add_argument("--mstcn", required=True, help="checkpoints/best_mstcn.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_filters", type=int, default=128)
    ap.add_argument("--no_viterbi", action="store_true", help="plain argmax instead of Viterbi decoding")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo).resolve()))
    from models.cnn_backbone import FrameBackbone
    from models.mstcn import MSTCN
    from utils.postprocess import viterbi_decode
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    backbone = FrameBackbone(num_classes=4)
    backbone.load_state_dict(torch.load(a.backbone, map_location="cpu"))
    backbone.to(device).eval()
    tcn = MSTCN(input_dim=2048, num_stages=2, num_layers=8, num_filters=a.num_filters, num_classes=4)
    tcn.load_state_dict(torch.load(a.mstcn, map_location="cpu"))
    tcn.to(device).eval()
    tfm = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    cap = cv2.VideoCapture(a.video)
    step = max(1, int(round(cap.get(cv2.CAP_PROP_FPS))))        # 1 frame per second, as in extract_features.py
    feats, i = [], 0
    with torch.no_grad():
        ok, frame = cap.read()
        while ok:
            if i % step == 0:
                x = tfm(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                feats.append(backbone(x, return_features=True)[1].cpu())
            ok, frame = cap.read()
            i += 1
        cap.release()
        f = torch.cat(feats).unsqueeze(0).to(device)               # [1, T, 2048]
        logits, _ = tcn(f)
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()     # [T, 4]
    preds = probs.argmax(1) if a.no_viterbi else viterbi_decode(probs, None, 4)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_s", "phase"] + [f"p_{p.replace(' ', '_')}" for p in PHASES])
        for t, (c, p) in enumerate(zip(preds, probs)):
            w.writerow([t, PHASES[c]] + [f"{v:.3f}" for v in p])
    segs, start = [], 0
    for t in range(1, len(preds) + 1):
        if t == len(preds) or preds[t] != preds[start]:
            segs.append((start, t, PHASES[preds[start]]))
            start = t
    print(f"{a.video}: {len(preds)} s at 1 FPS on {device}; wrote {a.out}")
    for s, e, ph in segs:
        print(f"  {s:4d}-{e:4d} s  {ph}")


if __name__ == "__main__":
    main()
