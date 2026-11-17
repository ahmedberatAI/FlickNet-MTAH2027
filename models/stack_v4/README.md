# Hyv path models (round 117, `src/model_v25.py`, research code `src/exp_stack.py` tag `Hyv`, `src/lab87/foldnets.py`)

The same width-128 stacker as `models/stack_v2/` (inputs, path model with the states home control, home free, away
control, away free, dead; training on the likelihood of the whole path; 2,000 steps of 60 s windows; equal provider
weight; random channel scales, dropped channels and a shift of the alive evidence), trained on more open data:

- Training data: the channels' held-out outputs on the open games only: 7 DFL matches (event-derived controlled labels),
  3 Metrica sample games (controlled labels, ball moved off its owner) and 20 SkillCorner matches (event-derived
  controlled labels), `sim.V6`; plus a copy of every DFL and Metrica half under `sim.V7m` (a ball that keeps moving in
  dead stretches and a low-passed ball path, every channel read on that ball) and a copy of every SkillCorner half under
  `sim.V6T`, with the same labels.
- `stack_hyv_s0.pt`, `stack_hyv_s1.pt`, `stack_hyv_s2.pt`: trained on all thirty games (and their copies) with seed
  indices 0, 1, 2 of the channels (their v10 and J1 seeds). `src/model_v25.py` draws the decision's samples from these
  three; the reference path still comes from `models/stack_v1/`, every channel and the decision are v23's.
- Held out by match, the decision on their samples scores higher than v23's in the lab and in the target's condition
  (channels and dead-ball models that never saw the provider) under `sim.V7` (round 117 in `docs/RAPOR.md`). They contain
  no match data and have never seen BAR-ATH.
