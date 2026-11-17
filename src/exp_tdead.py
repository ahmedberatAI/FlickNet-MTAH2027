"""Round 46 (docs/RAPOR.md): a sequence model of the dead ball. The multi-source dead model is a frame-level LightGBM on windowed features; a stoppage is a sequence (ball out,
players slow down, set-up, release). TD: small convolution + bidirectional GRU over the same 'base' features at 5 Hz, trained on DFL, de-glued Metrica and SkillCorner with
equal provider weight in every batch, with two auxiliary heads for stops and restarts to sharpen the boundaries. Seven folds, each holding out one DFL match, one Metrica game and a seventh of
SkillCorner; every half scored by models that never saw its match. AGGREGATE output only. Runs on the local GPU.

  python -I src/exp_tdead.py train SEED      the seven folds and the all-data model
  python -I src/exp_tdead.py infer SEED SRC  the all-data model on another source (st@v3, barath)
"""
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import bench  # noqa: E402

SOURCES = {"dfl": ("dfl@v3", "dfl", "native"), "metrica": ("metrica@v3g", "metrica@v3", "ctrl"), "sc": ("sc@v3", "sc@v3", "evctrl")}
OUT = os.path.join(bench.CACHE, "tdead")
if os.environ.get("TD_V6"):         # round 49: the same model on sim.V6's masks
    _v = os.environ.get("TD_VAR", "v6")
    SOURCES = {"dfl": ("dfl@%s" % _v, "dfl", "native"), "metrica": ("metrica@%sg" % _v, "metrica@v3", "ctrl"), "sc": ("sc@%s" % _v, "sc@v3", "evctrl")}
    OUT = os.path.join(bench.CACHE, os.environ.get("TD_OUT", "tdead6"))
    if os.environ.get("TD_V7"):     # round 110's V7: DFL and Metrica with the ball moving in dead time (V6M), SkillCorner's real tracking on V6
        SOURCES = {"dfl": ("dfl@v6m", "dfl", "native"), "metrica": ("metrica@v6mg", "metrica@v3", "ctrl"), "sc": ("sc@v6", "sc@v3", "evctrl")}
STEP = 5            # frames per model step (5 Hz)
WIN = 600           # steps per training window (2 minutes)
STEPS = 4000
BATCH = 24
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() and not os.environ.get("TD_CPU") else "cpu"


class Net(nn.Module):
    def __init__(self, n_in, width=128, drop=0.1):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(n_in, width, 5, padding=2), nn.GELU(), nn.Dropout(drop), nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.gru = nn.GRU(width, width, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.head = nn.Linear(2 * width, 3)

    def forward(self, x):
        h = self.front(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h)


def half_norm(x):
    """TD_NORM=1 (round 51): every feature centred and scaled by this half's own median and spread, so that the model cannot lean on a match's absolute tempo. Label-free and per half."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q1, q2, q3 = np.nanquantile(x, (0.25, 0.5, 0.75), axis=0)
        sd = np.nanstd(x, axis=0)
    scale = np.where(q3 - q1 > 1e-6, q3 - q1, np.where(sd > 1e-6, sd, 1.0))
    return (x - q2) / scale


def pool(x, step=STEP):
    """Mean over step frames, NaNs ignored."""
    n = len(x) // step
    x = x[:n * step].reshape(n, step, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.nanmean(x, axis=1)
    return half_norm(out) if os.environ.get("TD_NORM") else out


def targets(lab, step=STEP):
    """Per model step: dead share, mask (known frames), stop and restart heat (Gaussian of about 0.4 s)."""
    lab = np.asarray(lab)
    n = len(lab) // step
    y = lab[:n * step].reshape(n, step)
    known = y < 3
    dead = (y == 2).sum(1) / np.maximum(known.sum(1), 1)
    mask = known.mean(1) > 0.5
    d = (lab == 2).astype(np.int8)
    stops = (np.flatnonzero((d[1:] == 1) & (d[:-1] == 0) & (lab[1:] < 3) & (lab[:-1] < 3)) + 1) // step
    rest = (np.flatnonzero((d[1:] == 0) & (d[:-1] == 1) & (lab[1:] < 3) & (lab[:-1] < 3)) + 1) // step
    heat = np.zeros((n, 2), np.float32)
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    for col, idx in ((0, stops), (1, rest)):
        for i in idx:
            a, b = max(0, i - 6), min(n, i + 7)
            heat[a:b, col] = np.maximum(heat[a:b, col], kernel[a - i + 6:b - i + 6])
    return dead.astype(np.float32), mask, heat


def pff_rows():
    """PFF raw as a fourth provider (TD_PFF=1): the prepared per-half features ('base' columns, real ball gaps), its own D."""
    import gradient
    import pathlib
    names = bench.features("base", "dfl@v3")[0]["fk"]
    out = pathlib.Path(bench.ROOT) / "cache" / "pff_experiment" / gradient.VERSION / "features"
    rows, lens = [], []
    for match in gradient.matches():
        with np.load(gradient.CACHE / gradient.VERSION / ("%s.npz" % match)) as d:
            for per in d["periods"]:
                path = out / ("%s_h%d.pkl" % (match, int(per)))
                if not path.exists():
                    continue
                f = bench._load(str(path))
                x = pool(f["XF"][:, [f["fk"].index(k) for k in names]])
                rows.append(("pff:%s" % match, x.astype(np.float32),) + targets(np.asarray(d["h%d_labels" % int(per)])))
                lens.append(len(f["XF"]))
    return rows, lens


def load_all():
    data = {}
    for k, (fsrc, lsrc, conv) in SOURCES.items():
        fs = bench.features("base", fsrc)
        ls = bench.labels(lsrc, conv)
        gs = [b["group"] for b in bench.blocks(fsrc)]
        rows = []
        for f, l, g in zip(fs, ls, gs):
            x = pool(f["XF"])
            rows.append((g, x.astype(np.float32),) + targets(l))
        data[k] = rows
        print("  %s: %d halves" % (k, len(rows)), flush=True)
    if os.environ.get("TD_MIX"):         # round 112: + the V6M copy of every DFL / Metrica half (same groups, held out together)
        for k, src in (("dfl", "dfl@v6m"), ("metrica", "metrica@v6mg")):
            fs = bench.features("base", src)
            ls = bench.labels(SOURCES[k][1], SOURCES[k][2])
            gs = [b["group"] for b in bench.blocks(src)]
            data[k] = data[k] + [(g, pool(f["XF"]).astype(np.float32),) + targets(l) for f, l, g in zip(fs, ls, gs)]
            print("  %s + V6M copies: %d halves" % (k, len(data[k])), flush=True)
    if os.environ.get("TD_PFF"):
        data["pff"], _lens["pff"] = pff_rows()
        print("  pff: %d halves" % len(data["pff"]), flush=True)
    return data


_lens = {}


def score_pooled(net, mean, scale, x, n):
    """score_half on an already pooled half of n frames."""
    xx = np.nan_to_num((x - mean) / scale, nan=0.0)
    with torch.no_grad():
        out = net(torch.as_tensor(xx[None], dtype=torch.float32, device=DEVICE))[0, :, 0].cpu().numpy()
    grid = (np.arange(len(out)) + 0.5) * STEP
    z = np.interp(np.arange(n) + 0.5, grid, out)
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def fit(data, hold, seed):
    """Train on everything except the groups in hold; returns (net, mean, scale)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = {k: [r for r in rows if r[0] not in hold] for k, rows in data.items()}
    xs = np.concatenate([r[1][::5] for rows in pools.values() for r in rows])
    mean, scale = np.nanmean(xs, axis=0), np.nanstd(xs, axis=0) + 1e-6
    n_in = xs.shape[1]
    net = Net(n_in).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    keys = [k for k in pools if pools[k]]
    net.train()
    for step in range(STEPS):
        X, Yd, M, H = [], [], [], []
        for b in range(BATCH):
            rows = pools[keys[b % len(keys)]]
            _, x, d, m, h = rows[int(rng.integers(len(rows)))]
            a = int(rng.integers(0, max(1, len(x) - WIN)))
            xx = np.nan_to_num((x[a:a + WIN] - mean) / scale, nan=0.0)
            X.append(xx)
            Yd.append(d[a:a + WIN])
            M.append(m[a:a + WIN])
            H.append(h[a:a + WIN])
        L = min(len(v) for v in X)
        X = torch.as_tensor(np.stack([v[:L] for v in X]), dtype=torch.float32, device=DEVICE)
        Yd = torch.as_tensor(np.stack([v[:L] for v in Yd]), dtype=torch.float32, device=DEVICE)
        M = torch.as_tensor(np.stack([v[:L] for v in M]), dtype=torch.float32, device=DEVICE)
        H = torch.as_tensor(np.stack([v[:L] for v in H]), dtype=torch.float32, device=DEVICE)
        out = net(X)
        loss = (nn.functional.binary_cross_entropy_with_logits(out[:, :, 0], Yd, reduction="none") * M).sum() / M.sum().clamp(min=1)
        loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(out[:, :, 1:], H)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
    net.eval()
    return net, mean, scale


def score_half(net, mean, scale, xf):
    """P(dead) per frame of a half: the model at 5 Hz over the whole half, linearly interpolated to 25 fps."""
    x = pool(xf)
    xx = np.nan_to_num((x - mean) / scale, nan=0.0)
    with torch.no_grad():
        out = net(torch.as_tensor(xx[None], dtype=torch.float32, device=DEVICE))[0, :, 0].cpu().numpy()
    grid = (np.arange(len(out)) + 0.5) * STEP
    z = np.interp(np.arange(len(xf)) + 0.5, grid, out)
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def score_heads(net, mean, scale, xf):
    """Per frame: stop and restart heat of the model's auxiliary heads, as probabilities, interpolated to 25 fps (float16 pair)."""
    x = pool(xf)
    xx = np.nan_to_num((x - mean) / scale, nan=0.0)
    with torch.no_grad():
        out = net(torch.as_tensor(xx[None], dtype=torch.float32, device=DEVICE))[0, :, 1:].cpu().numpy()
    grid = (np.arange(len(out)) + 0.5) * STEP
    t = np.arange(len(xf)) + 0.5
    return np.stack([1 / (1 + np.exp(-np.interp(t, grid, out[:, c]))) for c in (0, 1)], 1).astype(np.float16)


def train(seed):
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    feats = {k: bench.features("base", SOURCES[k][0]) for k in SOURCES}
    res = {k: [[] for _ in data[k]] for k in SOURCES}
    heads = {k: [[] for _ in data[k]] for k in SOURCES}
    pf = sorted({r[0] for r in data.get("pff", [])})
    res_p = [[] for _ in data.get("pff", [])]
    for i in range(len(dfl)):
        t0 = time.time()
        hold = {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)]) | set(pf[i::len(dfl)])
        net, mean, scale = fit(data, hold, seed + i)
        for j, r in enumerate(data.get("pff", [])):
            if r[0] in hold:
                res_p[j].append(score_pooled(net, mean, scale, r[1], _lens["pff"][j]))
        for k in SOURCES:
            for j, r in enumerate(data[k]):
                if r[0] in hold:
                    res[k][j].append(score_half(net, mean, scale, feats[k][j]["XF"]))
                    heads[k][j].append(score_heads(net, mean, scale, feats[k][j]["XF"]))
        print("  seed %d fold %d done, %.0f s" % (seed, i, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()
    for k in SOURCES:
        out = [np.mean(np.stack(v), axis=0) for v in res[k]]
        assert all(len(v) for v in res[k])
        bench._dump(out, os.path.join(OUT, "held_%d_%s.pkl" % (seed, SOURCES[k][0].replace("@", "-"))))
        bench._dump([np.mean(np.stack(v), axis=0).astype(np.float16) for v in heads[k]], os.path.join(OUT, "heads_%d_%s.pkl" % (seed, SOURCES[k][0].replace("@", "-"))))
    if res_p:
        bench._dump([np.mean(np.stack(v), axis=0) for v in res_p], os.path.join(OUT, "held_%d_pff.pkl" % seed))
    net, mean, scale = fit(data, set(), seed + 100)
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mean=mean, scale=scale, n_in=len(mean)), os.path.join(OUT, "all_%d.pt" % seed))
    print("  seed %d all-data model done" % seed, flush=True)


def train_mix(seed):
    """Round 53: the mask's true behaviour around stops and restarts is unknown, so train on all three versions (V3, V6, V6B; domain randomisation over the simulator's uncertainty). Same folds, same steps; held-out outputs on each version."""
    seed = int(seed)
    out = os.path.join(bench.CACHE, os.environ.get("TD_OUT", "tdeadmix"))
    os.makedirs(out, exist_ok=True)
    lab = {"dfl": ("dfl", "native"), "metrica": ("metrica@v3", "ctrl"), "sc": ("sc@v3", "evctrl")}
    names = {"dfl": ("dfl@v3", "dfl@v6", "dfl@v6b"), "metrica": ("metrica@v3g", "metrica@v6g", "metrica@v6bg"), "sc": ("sc@v3", "sc@v6", "sc@v6b")}
    data = {}
    index = {}
    feats = {}
    for k, srcs in names.items():
        data[k] = []
        for fsrc in srcs:
            fs = bench.features("base", fsrc)
            ls = bench.labels(*lab[k])
            gs = [b["group"] for b in bench.blocks(fsrc)]
            feats[fsrc] = fs
            for j, (f, l, g) in enumerate(zip(fs, ls, gs)):
                index[(fsrc, j)] = g
                data[k].append((g, pool(f["XF"]).astype(np.float32),) + targets(l))
            bench._blocks.clear()
        print("  %s: %d halves" % (k, len(data[k])), flush=True)
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    res = {key: [] for key in index}
    for i in range(len(dfl)):
        t0 = time.time()
        hold = {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)])
        net, mean, scale = fit(data, hold, seed + i)
        for (fsrc, j), g in index.items():
            if g in hold:
                res[(fsrc, j)].append(score_half(net, mean, scale, feats[fsrc][j]["XF"]))
        print("  seed %d fold %d done, %.0f s" % (seed, i, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()
    for fsrc in {key[0] for key in index}:
        m = sum(1 for key in index if key[0] == fsrc)
        bench._dump([np.mean(np.stack(res[(fsrc, j)]), axis=0) for j in range(m)], os.path.join(out, "held_%d_%s.pkl" % (seed, fsrc.replace("@", "-"))))
    net, mean, scale = fit(data, set(), seed + 100)
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mean=mean, scale=scale, n_in=len(mean)), os.path.join(out, "all_%d.pt" % seed))
    print("  seed %d all-data model done" % seed, flush=True)


def lopo(seed):
    """Round 55: a whole provider unseen, the condition the target is in: each provider scored by models trained on the other two."""
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    feats = {k: bench.features("base", SOURCES[k][0]) for k in SOURCES}
    for i, k in enumerate(SOURCES):
        t0 = time.time()
        hold = {r[0] for r in data[k]}
        net, mean, scale = fit(data, hold, seed + 200 + i)
        out = [score_half(net, mean, scale, f["XF"]) for f in feats[k]]
        bench._dump(out, os.path.join(OUT, "lopo_%d_%s.pkl" % (seed, SOURCES[k][0].replace("@", "-"))))
        print("  seed %d provider %s done, %.0f s" % (seed, k, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()


_DV = os.environ.get("TD_DOSE_VAR", "v6m")
DOSE = {"dfl": "dfl@%s" % _DV, "metrica": "metrica@%sg" % _DV}       # round 110: V6M, the ball moving in dead time (round 113: v6s)
if os.environ.get("TD_DOSE_MAP"):     # round 116: {provider: dosed source}, e.g. V6T with a sigma per provider
    import json
    DOSE = json.loads(os.environ["TD_DOSE_MAP"])


def dose(seed):
    """Round 110: the folds of `train` (lab, held out by match) and of `lopo` (provider held out) again, each held-out
    DFL / Metrica half scored on its V6 features and on its V6M (dosed) features. Writes held / lopo files for both into
    OUT (set TD_OUT apart from the shipped folders)."""
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    dl = {k: [v] if isinstance(v, str) else list(v) for k, v in DOSE.items()}     # round 117: several dosed sources per provider
    feats = {k: bench.features("base", SOURCES[k][0]) for k in DOSE}
    dfeats = {(k, d): bench.features("base", src) for k in DOSE for d, src in enumerate(dl[k], 1)}
    res = {(k, d): [[] for _ in data[k]] for k in DOSE for d in range(len(dl[k]) + 1)}
    for i in range(len(dfl)):
        t0 = time.time()
        hold = {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)])
        net, mean, scale = fit(data, hold, seed + i)
        for k in DOSE:
            for j, r in enumerate(data[k]):
                if r[0] in hold:
                    res[(k, 0)][j].append(score_half(net, mean, scale, feats[k][j]["XF"]))
                    for d in range(1, len(dl[k]) + 1):
                        res[(k, d)][j].append(score_half(net, mean, scale, dfeats[(k, d)][j]["XF"]))
        print("  dose seed %d fold %d done, %.0f s" % (seed, i, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()
    for k in DOSE:
        for d, src in enumerate([SOURCES[k][0]] + dl[k]):
            bench._dump([np.mean(np.stack(v), axis=0) for v in res[(k, d)]], os.path.join(OUT, "held_%d_%s.pkl" % (seed, src.replace("@", "-"))))
    for i, k in enumerate(SOURCES):
        if k not in DOSE:
            continue
        t0 = time.time()
        hold = {r[0] for r in data[k]}
        net, mean, scale = fit(data, hold, seed + 200 + i)
        for src, fs in [(SOURCES[k][0], feats[k])] + [(s_, dfeats[(k, d)]) for d, s_ in enumerate(dl[k], 1)]:
            bench._dump([score_half(net, mean, scale, f["XF"]) for f in fs], os.path.join(OUT, "lopo_%d_%s.pkl" % (seed, src.replace("@", "-"))))
        print("  dose seed %d provider %s held out, %.0f s" % (seed, k, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()


V6SRC = {"dfl": "dfl@v6", "metrica": "metrica@v6g", "sc": "sc@v6"}


def dose7(seed):
    """Round 110's V7 (run with TD_V7=1): the lab folds and the provider-held-out models trained on V7 (V6M for DFL and Metrica,
    V6 for SkillCorner), every held-out half scored on its V6 features and, for DFL / Metrica, its V6M features; then the
    all-data model. Same folds and seeds as `train` / `lopo`."""
    seed = int(seed)
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    dfl = sorted({r[0] for r in data["dfl"]}, key=str)
    met = sorted({r[0] for r in data["metrica"]}, key=str)
    sc = sorted({r[0] for r in data["sc"]}, key=str)
    tests = {(k, src): bench.features("base", src) for k in SOURCES for src in [V6SRC[k]] + ([DOSE[k]] if k in DOSE else [])}
    res = {key: [[] for _ in fs] for key, fs in tests.items()}
    for i in range(len(dfl)):
        t0 = time.time()
        hold = {dfl[i], met[i % len(met)]} | set(sc[i::len(dfl)])
        net, mean, scale = fit(data, hold, seed + i)
        for (k, src), fs in tests.items():
            for j in range(len(fs)):             # the originals come first in data[k] (TD_MIX appends copies)
                if data[k][j][0] in hold:
                    res[(k, src)][j].append(score_half(net, mean, scale, fs[j]["XF"]))
        print("  v7 seed %d fold %d done, %.0f s" % (seed, i, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()
    for (k, src), v in res.items():
        bench._dump([np.mean(np.stack(x), axis=0) for x in v], os.path.join(OUT, "held_%d_%s.pkl" % (seed, src.replace("@", "-"))))
    for i, k in enumerate(SOURCES):
        t0 = time.time()
        hold = {r[0] for r in data[k]}
        net, mean, scale = fit(data, hold, seed + 200 + i)
        for (k2, src), fs in tests.items():
            if k2 == k:
                bench._dump([score_half(net, mean, scale, f["XF"]) for f in fs], os.path.join(OUT, "lopo_%d_%s.pkl" % (seed, src.replace("@", "-"))))
        print("  v7 seed %d provider %s held out, %.0f s" % (seed, k, time.time() - t0), flush=True)
        del net
        torch.cuda.empty_cache()
    net, mean, scale = fit(data, set(), seed + 100)
    torch.save(dict(state={k: v.cpu() for k, v in net.state_dict().items()}, mean=mean, scale=scale, n_in=len(mean)), os.path.join(OUT, "all_%d.pt" % seed))
    print("  v7 seed %d all-data model done" % seed, flush=True)


def load_all_model(seed):
    ck = torch.load(os.path.join(OUT, "all_%d.pt" % int(seed)), map_location="cpu", weights_only=False)
    net = Net(ck["n_in"]).to(DEVICE)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["mean"], ck["scale"]


def held(src, seeds=(20261104, 20261105, 20261106)):
    """Mean held-out P(dead) per half over the seeds."""
    runs = [bench._load(os.path.join(OUT, "held_%d_%s.pkl" % (s, src.replace("@", "-")))) for s in seeds]
    return [np.mean(np.stack([r[i] for r in runs]), axis=0) for i in range(len(runs[0]))]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "mix":
        train_mix(sys.argv[2])
    elif sys.argv[1] == "dose":
        dose(sys.argv[2])
    elif sys.argv[1] == "dose7":
        dose7(sys.argv[2])
    elif sys.argv[1] == "lopo":
        lopo(sys.argv[2])
    else:
        raise SystemExit(__doc__)
