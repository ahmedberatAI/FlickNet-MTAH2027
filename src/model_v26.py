"""v26 candidate (round 132, docs/RAPOR.md): v25 with part of the decision's samples drawn from stackers trained on the
loose reading of possession. Not the default: the switch is the team's decision.

The official documents say the ground truth is what the corrected tracking names as the owning team, and that the public
five minutes show what "owning" means. Round 58 compared the label structure of those minutes with the open readings:
short possessions squeezed between two spells of the other team come at 0.99 per live minute there, at 0.20-0.28 under
the strict readings v25's stackers were trained on (DFL event-controlled, Metrica controlled, SkillCorner
event-controlled) and at 0.69-1.38 under the loose ones (DFL "any event within 1 s", Metrica "BALL LOST counts",
SkillCorner player possessions). Round 132: an open five-minute window shows the public window's structure in 1.0% of
the cases under the strict family and in 17.0% under the loose one, so the loose reading is the likelier meaning of the
labels (P about 0.8 after halving the log likelihood ratio).

The loose stackers (round 121, tag HyLv; models/stack_v5/) are Hyv's recipe, rows, folds and seeds (models/stack_v4/)
with the loose readings as training labels. Here the decision's posterior is a mixture: every strict stacker enters
`reps[0]` times and every loose stacker `reps[1]` times (stack_decode.decide draws the same number of paths from each
member); the reference path (v22's width-64 stackers), every channel, the dead-ball evidence and the decision's rule are
v25's. No label of the match is read anywhere; the public window chose nothing: the share (one strict copy to three
loose ones, M75v) was chosen on open data by round 132's pre-registered rule, the arm with the highest worst-cell
expected gain over v25 when the loose reading is given the probability 0.8. The decision's threshold lambda is 0.4 instead
of 0.5 (the theory value F* / 2 for a posterior whose spells match at about 0.78; the pre-registered addendum).

Frames before the second kick-off (official FAQ) stay dead. Deterministic on the CPU.
"""
import os

import numpy as np

import atd
import bench
import model_v7
import model_v13
import model_v20
import model_v21
import model_v22
import model_v23
import model_v25
import stack_decode as sd
import stack_model as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "v26")
NETS_L = tuple(os.path.join(ROOT, "models", "stack_v5", "stack_hylv_s%d.pt" % s) for s in (0, 1, 2))
DEFAULT = dict(model_v23.DEFAULT, reps=(1, 3), lam=0.4)   # M75v: a quarter of the samples strict, three quarters loose; lambda F* / 2


def export_nets():
    """models/stack_v5/: the loose-reading stackers' all-data nets (src/lab87/foldnets.py, tag HyLv of round 121) in stack_model's format."""
    import torch
    os.makedirs(os.path.dirname(NETS_L[0]), exist_ok=True)
    for s, dst in enumerate(NETS_L):
        if not os.path.isfile(dst):
            ck = torch.load(os.path.join(ROOT, "cache", "lab87", "foldnets", "HyLv_%d_all.pt" % s), map_location="cpu", weights_only=False)
            torch.save(dict(d_in=ck["d_in"], e=ck["e"], state=ck["state"]), dst)


def ensure_loose(f):
    """Per half and seed: the loose stackers' emissions (T, 5) and moves (5, 5), on v22's inputs; on the CPU (as model_v25.ensure_wide)."""
    path = os.path.join(OUT, "loose.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    import torch
    torch.set_num_threads(4)
    if not all(os.path.isfile(p) for p in NETS_L):
        export_nets()
    a_all = model_v20.alive_evidence(f)
    gru = bench._load(model_v13.ensure_gru())
    bfs = model_v21.ensure_bf()
    bench._barath_inputs()
    res = {}
    for s in range(3):
        v10 = model_v22._v10(s)
        j1 = model_v22._j1(s)
        net = sm.load(NETS_L[s])
        for b in bench.blocks("barath"):
            q_b, a_b = bfs[b["a"]]
            x = sm.inputs(b, v10[b["a"]], gru[b["a"]][1], j1[b["a"]], a_all[b["a"]:b["b"] + 1], q_b, a_b)
            with torch.no_grad():
                oh, oa = net(torch.as_tensor(x[None], dtype=torch.float32))
                d = 0.5 * (oh[..., 2] + oa[..., 2])
                e = torch.stack([oh[..., 0], oh[..., 1], oa[..., 0], oa[..., 1], d], -1)[0].numpy()
                A = sm.moves(net.t).numpy()
            res.setdefault(b["a"], []).append((e.astype(np.float32), A.astype(np.float32)))
    os.makedirs(OUT, exist_ok=True)
    bench._dump(res, path)
    return res


def predict(f, params=None):
    p = dict(DEFAULT)
    if params:
        p.update(params)
    rs, rl = (int(v) for v in p["reps"])
    a_all = model_v20.alive_evidence(f)
    mem = model_v22.ensure_members(f)
    wide = model_v25.ensure_wide(f)
    loose = ensure_loose(f)
    sp = model_v7.spec(p)
    labels = np.full(f["n"], 3, np.int8)
    for h, a, e in atd.HALVES:
        ms = mem[a]
        n = min(min(len(x[0]), len(x[1]), len(x[2])) for x in ms)
        n = min(n, e + 1 - a)
        lq = np.mean([model_v22.W_Q * sm.logit(x[1][:n]) + (1 - model_v22.W_Q) * sm.logit(x[0][:n]) for x in ms], axis=0)
        q = np.clip(1 / (1 + np.exp(-lq)), 1e-4, 1 - 1e-4)
        al = (1 - model_v22.W_A) * a_all[a:a + n] + model_v22.W_A * np.mean([x[2][:n] for x in ms], axis=0)
        S = np.stack([al / 2 + np.log(q), al / 2 + np.log(1 - q), -al / 2], 1)
        if a == atd.HALVES[1][1]:
            S[:atd.KICKOFF_2ND - a, 2] += 1e6
        ref = sp.decode(S).astype(np.int8)
        members = list(wide[a]) * rs + list(loose[a]) * rl          # round 121's order: the strict stackers' copies, then the loose ones'
        out = sd.decide(members, ref, p["lam"], int(p["samples"]), int(p["seed"]) + 1000 * h)
        if p.get("keep_dead"):
            out = sd.keep_reference_dead(out, ref)
        if a == atd.HALVES[1][1]:
            out[:atd.KICKOFF_2ND - a] = 2
        labels[a:a + len(out)] = out
    return labels
