# Loose-reading path models (round 121, `src/model_v26.py`, research code `src/exp_stack.py` tag `HyLv`, `src/lab87/loose_readings.py`)

The same width-128 stacker, inputs, rows, folds and seeds as `models/stack_v4/` (Hyv: the open games under `sim.V6`
plus a `sim.V7m` copy of every DFL and Metrica half and a `sim.V6T` copy of every SkillCorner half). The only
difference is the training labels, a looser reading of "possession" on the same open games:

- DFL: the tracking's own possession flag, where a team's spell after the other team's spell is kept when that team
  has any event within the spell (one second of margin) or a control event; otherwise it is merged back.
- Metrica sample games: the team of the last controlling event, where such a spell is kept when the team passes or
  shoots inside it, holds or carries the ball for a second, or loses it (a BALL LOST event counts as having had it).
- SkillCorner: the team of the dynamic events' player possessions.

Under these readings a short possession squeezed between two spells of the other team comes about once per live
minute, against 0.2-0.3 under the strict readings `models/stack_v4/` was trained on.

- `stack_hylv_s0.pt`, `stack_hylv_s1.pt`, `stack_hylv_s2.pt`: trained on all thirty games (and their copies) with
  seed indices 0, 1, 2 of the channels. `src/model_v26.py` mixes their path samples with `models/stack_v4/`'s in the
  metric's Bayes decision.
- They contain no match data and have never seen BAR-ATH.
