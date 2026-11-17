# Possession labelling for BAR-ATH — short technical document

*More than a Hack 2027, Phase 1. Team: [team name] ([members]). Repository: [public URL].*

*Submitted file: `[team name].csv` (199,319 rows, MD5 `aeec84984d731d171526a8a746013479`). It is a byte copy of
`submissions/candidate_v26.csv`, which `python src/make_submission.py` regenerates from the provided data. The full
document, with every component, experiment and number, is `docs/TECHNICAL_DOCUMENT.md` in the repository.*

## 1. Approach

The ranking metric counts possessions (spell F1 over IoU 0.3–0.7), not frames, and the organisers' own baselines show
that the two disagree. The pipeline is therefore built to get the *sequence* of possessions right: several per-frame
evidence channels that look at the tracking in different ways, a path model that fuses them, and a decision that
maximises the score the labelling is expected to get.

1. **Inputs.** The tactical ATD (players and ball). The broadcast ATD is aligned to the tactical timeline by a constant
   lag per half and fills ball gaps where the two cameras agree. All features are collective (team level): the tracking
   has no player identities.
2. **Dead ball.** Four pieces of evidence: a small rule model on player tempo and ball state, a LightGBM classifier
   trained on three open providers, a sequence model whose inputs are scaled by each half's own level, and whether the
   aligned broadcast shows the pitch (replays and close-ups are almost always dead time).
3. **Four team channels**, deliberately different views of the same tracking:
   a LightGBM classifier on about 150 team-level features; a bidirectional GRU on per-frame features; a ball-centred
   short-possession specialist (a GRU over the players nearest the ball at 25 Hz, with heads for control, touch, gain and
   loss); and a ball-free channel that reads only the players (a set encoder and a 60 s bidirectional GRU), whose errors
   are independent of the ball-based channels.
4. **Stacker.** A small path model at 5 Hz on the channels' held-out outputs: convolutions and a bidirectional GRU give
   emissions for five states (control and free per team, dead) with learned transitions, and possession passes to the
   other team only through its control state. It is trained with the exact path likelihood, with augmentation, and with
   copies of the open games degraded like the target's tracker.
5. **Decision.** Each half is labelled by the Bayes decision for the metric: 3,000 whole paths are sampled from the
   stackers' posterior; the candidates are every sampled segment and a duration-aware Viterbi path; a possession is kept
   when its mean chance of matching a true possession over the five IoU thresholds exceeds 0.4; dynamic programming
   picks the best full tiling of the half.
6. **The reading of possession is part of the decision.** A quarter of the sampled paths come from stackers trained on
   a strict reading of "controlled possession" and three quarters from stackers trained on a looser one (section 4).

**Why this rather than something else.** No label of the provided match may be used, so everything has to transfer
from other matches. Each step above was kept only if it improved the worst case over held-out matches of other
providers by at least 0.005 (from the 24th round on under a rule written down before the result; about 150 rounds of
experiments, most of them negative). The open tracking is first degraded by a simulator of the automatic tracker (ball dropouts, a low-passed
ball, smooth position noise, fragmented player tracks) whose rates come from label-free aggregate statistics of the
provided tracking. Per-frame decisions flicker, and the spell metric charges every false change (the official
baseline, nearest player with smoothing, scores 0.2433 on the full match). The steps that mattered most were the
broadcast view as dead-ball evidence, simulating the target tracker's behaviours in training, the ball-free channel
with the stacker, deciding for the metric instead of decoding the most likely path, and treating the label rule as
uncertain.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players, ball) | Main input. |
| Broadcast ATD and homography | Used: ball gaps filled where both cameras agree; "no valid homography" as a dead-ball cue. |
| Tactical homography | Read, not used as evidence (the ATD is already in pitch coordinates). |
| Smart Tagging | Dropped as evidence (evaluated as team and restart evidence; its phases are too coarse to place a change of possession). Its set pieces serve only as a label-free sanity check of our restarts. |
| Tactical video | Dropped: ball detection on the wide view (pretrained WASB, a fine-tuned YOLO) did not reach a useful quality. |
| Broadcast video | Dropped: an off-the-shelf ball-action spotter agrees with the model inside long possessions and is silent at contested moments; camera framing as possession evidence lost when fused (tested on SkillCorner, which has the camera). |
| Audio | Dropped: a referee-whistle detector could only be supported by the public window, and player names in the commentary were sparse and biased to one team. |
| Public 5-minute ground truth | Pipeline check only. Three counts of its label structure informed the reading of the label rule (section 4); no model was scored on it to choose anything. One first-day trace remains: the gate for filling ball gaps from the broadcast (full document, section 6). |

## 3. Training data

Every labelled dataset is from other matches; none comes from the provided game, and no frame of it was labelled by hand.

| Dataset | Source and licence | Labels: who and what | Amount | Used for |
|---|---|---|---|---|
| IDSSE / DFL open data (Bassek et al. 2025) | figshare 28196177, CC BY 4.0 | The provider: official event data and the tracking's per-frame possession and ball-status flags | 7 matches | Team classifier, bidirectional GRU (with PFF), specialist fine-tuning, dead-ball models, stackers; validation |
| Metrica Sports sample data | github.com/metrica-sports/sample-data | The provider's event data | 3 games | Team classifier, specialist fine-tuning, dead-ball models, stackers; validation |
| SkillCorner open data | github.com/SkillCorner/opendata, MIT | The provider's phases of play and player possessions on broadcast tracking | 20 matches | Dead-ball models, stackers; validation of the team channels (never in their training) |
| PFF FC / Gradient Sports, FIFA World Cup 2022 | Provider's access form | Human-labelled events | 64 matches | Bidirectional GRU (with DFL), specialist pre-training, ball-free channel; independent test of the dead-ball evidence |
| SoccerTrack v2 | CC BY 4.0 | The provider's per-frame ball status and events | 8 labelled matches | Independent test only (no submitted model was trained on it) |

Possession labels are derived from these annotations in two readings (section 4). Trained weights are shipped in
`models/`; the open data itself is not redistributed and is not needed to regenerate the submission.

## 4. Possession rules implemented

- **Dead ball** is always `D` and never a possession; it splits possessions. The frames of the second half before its
  kick-off are `D` (official FAQ). After a stoppage the team is decided from the first seconds of play.
- **A change of possession needs control by the other team.** What counts as control is read two ways on the open
  data. *Strict:* a completed pass, a shot, or the ball kept for about a second; a touch, a deflection or an aerial
  duel without subsequent control is merged back into the previous team's possession. *Loose:* a ball won and lost at
  once also counts (DFL's possession flag kept when that team has an event within a second; Metrica's events with a
  ball lost counted as having had the ball; SkillCorner's player possessions).
- **Which reading?** The challenge says the ground truth comes mechanically from corrected tracking, that the written
  definition is not a specification, and that the five public minutes show what the labels mean. Their label structure
  (17 possessions in 4.02 live minutes, four of them short and squeezed between two possessions of the other team)
  appears in 1.0% of the open five-minute windows under the strict readings and in 17.0% under the loose ones. The
  decision therefore gives the loose reading the weight 0.75. Only these counts were used; no model was scored on the
  window to choose anything.
- **Loose or contested ball:** possession stays with the previous team unless the evidence for the other team is
  strong enough for a possession to be expected to match (the path model's "free" states and the decision's threshold).
- **Challenges, deflections, aerial duels:** no hand-written rule; a short possession of the other team is emitted
  when the mixture of the two readings gives it a chance of matching above the threshold.
- **Last touch before the ball leaves the pitch:** no rule turns it into a possession; the model emits one only where
  the evidence alone supports it (section 6).

## 5. Validation and results

Held-out external matches only: DFL leave-one-match-out, Metrica leave-one-game-out, SkillCorner; at least three
seeds, a paired bootstrap over matches, the worst case over label conventions, and rules fixed before the results. From
November on, every candidate is also scored in the *target's condition* (the scored provider withheld from every
channel and dead-ball model) and on inputs simulated like the target's tracking. SoccerTrack v2 and the raw PFF
matches stay untouched tests. Spell F1 of the submitted model on inputs simulated like the target's tracking:

| Cell | Strict reading | Loose reading |
|---|---|---|
| DFL, lab (other matches of the provider seen) | 0.776 | 0.760 |
| DFL, target condition | 0.782 | 0.748 |
| Metrica, lab | 0.819 | 0.806 |
| Metrica, target condition | 0.815 | 0.804 |

The previous model, trained for the strict reading only, scores 0.82–0.90 on the strict reading and 0.69–0.72 on the
loose one; the submitted model's score hardly depends on the reading. The official baseline's published full-match
score is 0.2433.

## 6. Limitations

- **The label rule is not known.** If the strict reading is the true one, the submitted model gives up 0.04–0.08
  against a strict-only model; if the loose one is, it gains 0.05–0.09.
- **Short possessions between two possessions of the other team.** Under the loose reading the model finds only
  about a third to a half of them. The automatic tracking (ball missing on 41% of the graded frames, its turns rounded)
  does not show which player touched the ball; reliable touch recognition from video would be needed, and we did not
  obtain it.
- **Last touch before a stoppage.** If the ground truth counts it (for example a defender's touch before a corner), the
  model misses those possessions. On open data, even when such a possession is known to exist, a rule-based insertion
  meets only 30–41% of the IoU thresholds, because it lasts about a second and the stoppage start is uncertain by as much.
- **Everything is validated on other providers.** With the evaluated provider withheld from the stacker as well, the
  scores are 0.02–0.03 lower on DFL and SkillCorner, so the table above may be optimistic for the provided match. The
  Metrica sample games' ball is generated from their events, which flatters ball-based evidence.
- **Dead-ball edges.** Where it is unsure at the edge of a stoppage the decision prefers `D`, which costs a little
  frame accuracy (the tie-breaking metric).
- **With more time:** touch and ball recognition from the two videos, and a published definition of the label rule.

## 7. Reproduction

```bash
# Python 3.12; pinned packages in requirements.txt; no GPU is used
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt        # Linux / macOS: .venv/bin/python
git clone https://github.com/metrica-sports/more-than-a-hack-2027 external/more-than-a-hack-2027
# unpack the match bundle into data/{tactical,broadcast} (or set MTAH_DATA_DIR)
.venv/Scripts/python.exe src/make_submission.py                             # -> submissions/candidate_v26.csv
python external/more-than-a-hack-2027/evaluation/validate_submission.py submissions/candidate_v26.csv
```

The run takes a few minutes on a CPU and is deterministic (fixed seeds, LightGBM in deterministic mode, no GPU). From
a clean checkout and an empty cache it writes the submitted file, MD5 `aeec84984d731d171526a8a746013479`, on Windows 11
(Python 3.12.13) and on Ubuntu 24.04 (Python 3.12.3), both x86-64; it was not tested on ARM. All trained weights are
shipped in `models/`; how each was trained is in section 10 of the full document.
