"""End-to-end, deterministic: provided data -> submission CSV.

    .venv/Scripts/python.exe src/make_submission.py      # the submission: model_v26 -> submissions/candidate_v26.csv
    .venv/Scripts/python.exe src/make_submission.py --model model_v3 --out submissions/baseline_v3.csv

Steps: parse ATD (cached), align the broadcast (fixed per-half lags), build per-frame features
with the broadcast ball filling tactical gaps, score H/A/D evidence, min-duration Viterbi per half (v22 on: then the metric's Bayes decision on the stackers'
posterior, src/stack_decode.py),
write and validate the file, print label-free full-match diagnostics. The public window's score (a pipeline check that
never chooses a version, official rules) is printed only with --public.
"""
import argparse
import importlib
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import broadcast  # noqa: E402
import evaluate as ev  # noqa: E402
import features  # noqa: E402
import proxy  # noqa: E402

CACHE = os.path.join(ROOT, "cache")


def merged_features(refresh=False):
    path = os.path.join(CACHE, "features_merged.pkl")
    if os.path.isfile(path) and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    os.makedirs(CACHE, exist_ok=True)
    b = broadcast.build()
    override = np.where((b["b_seen"] & b["b_live"])[:, None], b["b_ball"], np.nan)
    f = features.build(ball_override=override)
    f.update(b)
    with open(path, "wb") as fh:
        pickle.dump(f, fh, protocol=4)
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_v26")
    ap.add_argument("--params", default="{}", help="JSON overrides of the model's DEFAULT")
    ap.add_argument("--out", default=os.path.join(ROOT, "submissions", "candidate_v26.csv"))
    ap.add_argument("--refresh", action="store_true", help="rebuild the feature cache")
    ap.add_argument("--public", action="store_true", help="also print the public window's score (a pipeline check only)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    f = merged_features(args.refresh)
    if args.model == "model_v5":
        import train_dfl
        train_dfl.ensure_barath_probs(args.refresh)
    model = importlib.import_module(args.model)
    labels = model.predict(f, json.loads(args.params))
    path = ev.write(labels, args.out)
    print("wrote", path)
    if args.public:
        print("public window :", ev.fmt(ev.score(labels)))
    print("full match    :", ev.fmt(ev.diagnostics(labels)))
    print("proxies       :", ev.fmt(proxy.proxies(labels, f)))
