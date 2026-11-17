"""v25 candidate (round 117, docs/RAPOR.md): v24 with the decision's samples drawn from stackers that have also seen how the
target's tracker follows the ball in live play. Not the default: the switch is the user's decision.

Rounds 109-113: the ball moves in the target's dead stretches; Hym (v24) learned it from dosed copies of every DFL and
Metrica half. Round 115: the target's tracker rounds the ball's turns (2.85 sharp turns per live minute against 10-66 in
the open input); round 116: a low-pass of the ball's seen runs (sim's post_smooth_b) costs v23 0.011 on Metrica out of
provider, mostly through the team channels. Round 117: V7 = both behaviours together. Hyv = Hyw's recipe, data and seeds
plus a V7m copy of every DFL and Metrica half (synthetic moving dead ball and rounded turns, every channel read on the
rounded ball) and a V6T copy of every SkillCorner half. Here the three width-128 stackers of v23 (models/stack_v2/) are
replaced by Hyv's all-data nets (models/stack_v4/); the reference path (v22's width-64 stackers), every channel, the
dead-ball evidence and the decision are v23's.

Frames before the second kick-off (official FAQ) stay dead. Every model was trained on open data only; nothing here reads
a label of the match. Deterministic on the CPU.
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
import stack_decode as sd
import stack_model as sm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "cache", "v25")
NETS = tuple(os.path.join(ROOT, "models", "stack_v4", "stack_hyv_s%d.pt" % s) for s in (0, 1, 2))
DEFAULT = dict(model_v23.DEFAULT)


def export_nets():
    """models/stack_v4/: Hyv's all-data nets (src/lab87/foldnets.py, tag Hyv of round 117) in stack_model's format."""
    import torch
    os.makedirs(os.path.dirname(NETS[0]), exist_ok=True)
    for s, dst in enumerate(NETS):
        if not os.path.isfile(dst):
            ck = torch.load(os.path.join(ROOT, "cache", "lab87", "foldnets", "Hyv_%d_all.pt" % s), map_location="cpu", weights_only=False)
            torch.save(dict(d_in=ck["d_in"], e=ck["e"], state=ck["state"]), dst)


def ensure_wide(f):
    """Per half and seed: Hyv's emissions (T, 5) and moves (5, 5), on v22's inputs; on the CPU."""
    path = os.path.join(OUT, "wide.pkl")
    if os.path.isfile(path):
        return bench._load(path)
    import torch
    torch.set_num_threads(4)
    if not all(os.path.isfile(p) for p in NETS):
        export_nets()
    a_all = model_v20.alive_evidence(f)
    gru = bench._load(model_v13.ensure_gru())
    bfs = model_v21.ensure_bf()
    bench._barath_inputs()
    res = {}
    for s in range(3):
        v10 = model_v22._v10(s)
        j1 = model_v22._j1(s)
        net = sm.load(NETS[s])
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
    a_all = model_v20.alive_evidence(f)
    mem = model_v22.ensure_members(f)
    wide = ensure_wide(f)
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
        out = sd.decide(list(wide[a]), ref, p["lam"], int(p["samples"]), int(p["seed"]) + 1000 * h)
        if p.get("keep_dead"):
            out = sd.keep_reference_dead(out, ref)
        if a == atd.HALVES[1][1]:
            out[:atd.KICKOFF_2ND - a] = 2
        labels[a:a + len(out)] = out
    return labels
