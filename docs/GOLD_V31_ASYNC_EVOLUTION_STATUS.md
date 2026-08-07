# Pokémon TCG AI Battle — Gold V3.1 Async Evolution Pipeline

Last updated: 2026-08-08 JST
Status: running locally on the Doraemon cluster; no Kaggle submission is authorized or claimed

## Executive summary

Gold V3.1 is an asynchronous agent-evolution pipeline for two complementary specialists:

- Garchomp specialist
- Grimmsnarl anchor

Candidate production no longer waits for tournament evaluation. The training side continuously generates immutable policy or deck candidates. An independent evaluator consumes the queue and promotes only candidates that pass staged, multi-opponent, two-seat gates against the current champion. A failed candidate cannot overwrite the champion, and an unverified candidate cannot become the parent of the next generation.

The current system is an engineering and experimental pipeline, not evidence that a gold-medal agent already exists.

## Why the previous approach stalled

The redesign incorporates the most consistent findings from recent public notebooks, discussions, replays, and our own local experiments:

1. Higher BC accuracy is not equivalent to higher head-to-head win rate.
2. Changing only the 60-card deck can move performance more than changing a checkpoint.
3. A scalar value head can recognize dangerous positions but does not identify which action caused the danger.
4. Reward shaping can reduce selected error classes without improving final win rate.
5. Clone accuracy can be extremely high while head-to-head performance regresses.
6. Counterfactual engine branches are more useful for action selection than a state-only value estimate.
7. Suspected gains require hundreds of games and an independent reproduction.
8. Serial “train, stop, evaluate for hours, resume” wastes the training servers and creates low iteration throughput.

Gold V3.1 therefore separates representation experiments, candidate generation, evaluation, and champion mutation.

## Agent inputs and learning targets

The V3.1 model supports the following leakage-safe inputs:

- Structured card and attack identity encoding.
- Causal Transformer history of prior pre-action states and selected-option summaries.
- Exact own remaining-card multiset derived from the submitted 60-card deck and visible own zones.
- Key-card draw-probability summaries.
- Opponent deck belief derived only from public active, bench, discard, attachments, and frozen deck prototypes.
- Explicit deck conditioning for both the learner and opponent archetype belief.

Hidden opponent hand and prize-card identities are never used, even if malformed replay observations contain them.

The learning stack contains:

- Deck-specialized behavior cloning initialization.
- Frozen-league self-play rollouts.
- Multi-environment PPO.
- Search-policy distillation from engine branches.
- Loss-prioritized counterfactual action targets.
- Dueling action value decomposition, `Q(s,a) = V(s) + A(s,a)`.
- Conservative Q reranking with uncertainty, margin, and maximum override-rate limits.

Action Q is attached at inference only when its validation rows and MAE meet the configured calibration thresholds. The final tournament gate still evaluates the complete packaged agent.

## Asynchronous architecture

```text
                       immutable candidate queue
┌──────────────────┐  checkpoint + deck + Q hashes  ┌──────────────────┐
│ producer          │ ─────────────────────────────▶ │ evaluator         │
│                  │                                 │                  │
│ self-play         │                                 │ 20-game smoke    │
│ PPO               │                                 │ 200-game screen  │
│ branch search     │                                 │ 400-game confirm │
│ Q / advantage     │                                 │ independent 400  │
│ deck mutation     │                                 │                  │
└────────┬─────────┘                                 └────────┬─────────┘
         │ frozen parent/league                                │ compare-and-swap
         │                                                     ▼
         └─────────────────────────────── champion registry + frozen league
```

Each specialist has one persistent producer and one persistent evaluator.

### Producer guarantees

- Reads an atomic snapshot of the current champion and league at generation start.
- Never changes that snapshot while producing the candidate.
- Generates self-play, PPO, search-distilled, and dueling-Q policy candidates.
- Generates one legal deck-mutation candidate every fourth generation.
- Rotates deterministic PBT variants: conservative, ultra-conservative, controlled exploration, and league-robust.
- Writes an immutable candidate manifest containing checkpoint, deck, Q, and composite hashes.
- Deduplicates packages before adding them to the queue.
- Continues while evaluation is busy, up to a bounded backlog of eight candidates per specialist.

### Evaluator guarantees

- Uses the current champion at evaluation start, not blindly the candidate's training parent.
- Resumes an interrupted `evaluating` candidate from existing shard files.
- Prioritizes an interrupted evaluation first, then candidates trained from the current champion.
- Discards candidates more than two champion versions stale.
- Runs exact `20 -> 200 -> 400 -> independent 400` staged gates.
- Requires zero failures and controls head-to-head score, Wilson lower bound, public-opponent delta, worst-matchup regression, and seat gap.
- Updates the champion through versioned compare-and-swap; a stale result cannot overwrite a newer champion.
- May retain a safe but rejected behavioral variant in the frozen league, but never as champion.

## Staged promotion protocol

| Stage | Games | Purpose |
|---|---:|---|
| Smoke | 20 | Reject crashes, illegal behavior, catastrophic regressions, and obviously weak candidates quickly. |
| Screen | 200 | Establish an initial multi-opponent and two-seat signal. |
| Confirm | 400 | Apply the formal head-to-head, public-pool, worst-matchup, and seat-balance thresholds. |
| Independent replication | 400 | Repeat the formal test with an independent seed table. |

Candidates failing an earlier stage do not consume the later stages. A representation experiment is considered reproducible only when independently trained seeds both pass.

## Deck-policy evolution

Policy and deck search use coordinated, auditable updates:

- Policy generations preserve the current champion deck while updating the policy.
- Every fourth generation creates a legal 60-card mutation with at most three swaps.
- Deck candidates retain the exact champion checkpoint and calibrated Q package, changing only the deck.
- Deck and policy candidates enter the same local tournament queue.
- A promoted deck becomes the deck used by subsequent policy generations.

This provides continuous joint evolution without changing both variables inside one unidentifiable training update.

## Server responsibilities

| Server | Primary role |
|---|---|
| doraemon03 | Coordinator, Grimmsnarl self-play workers, local training selection, and service supervision. |
| doraemon20 | Garchomp self-play workers and GPU training. |
| doraemon15 | Strict tournament evaluation under the shared global evaluation lock. |

All servers use the shared `/homes/lzhang/pocketmon` disk for data and artifacts. Code for this pipeline is isolated at `/homes/lzhang/pocketmon_train_goldv31`.

## Durable state and recovery

Per-specialist run roots:

- `/homes/lzhang/pocketmon/results/continuous_rl/gold_v31_async_garchomp`
- `/homes/lzhang/pocketmon/results/continuous_rl/gold_v31_async_grimmsnarl`

Important files:

- `champion.json`: versioned current champion and hashes.
- `league.json`: bounded frozen training population.
- `producer_state.json`: active generation and crash-recovery state.
- `evaluator_state.json`: active candidate and evaluation state.
- `generation_*/working.json`: immutable parent and league snapshot.
- `generation_*/candidate.json`: immutable candidate identity and hashes.
- `generation_*/lifecycle.json`: queued, evaluating, promoted, rejected, duplicate, or stale status.
- `producer_events.jsonl`, `evaluator_events.jsonl`, `champion_history.jsonl`: append-only audit logs.
- `STOP`: graceful service stop marker.

Producer and evaluator use separate exclusive role locks. Restarting either service does not require stopping the other.

## Deployment snapshot

The asynchronous pipeline was deployed from commit:

```text
539b7ea2444dc6253cab568705a8ac57ac9a9704
```

Initial service PIDs at deployment:

- Garchomp producer: `95874`
- Garchomp evaluator: `95875`
- Grimmsnarl producer: `95877`
- Grimmsnarl evaluator: `95878`

At the deployment check:

- Both producers were actively collecting generation-1 self-play.
- Garchomp had 12 rollout workers on doraemon20.
- Grimmsnarl had 4 rollout workers on doraemon03.
- Both evaluators were independently waiting for their first immutable candidate.
- Startup error scans were empty.
- The initial champions remained the frozen Gold V3 G0 packages.

The previous synchronous Grimmsnarl coordinator was stopped before it acquired the evaluation lock. The previous Garchomp coordinator was given a graceful STOP and allowed to finish its already-running gate so its completed games were not discarded.

## Validation evidence

- Full regression suite: `156/156` tests passed.
- Deterministic CUDA forward, backward, and checkpoint smoke passed.
- Two-seat packaged-agent smoke had zero load, inference, illegal-action, and fallback errors.
- Candidate identity changes when either checkpoint, deck, or action-Q hash changes.
- Unit tests cover queue recovery, queue ordering, bounded backlog, deck cadence, champion compare-and-swap, configuration stages, and delivery of `--dueling-advantage` to the Q trainer.

## Current limitations and open risks

- No candidate has yet passed the complete local promotion protocol in this asynchronous line.
- Local engine evaluation remains the bottleneck and is intentionally serialized by the global evaluation lock.
- PPO is asynchronous with evaluation but remains synchronous within each training generation; IMPALA/V-trace is not yet the production optimizer.
- Public replay rating and matchup distributions remain biased, so training continues to use recency weighting and opponent/deck stratification.
- A low offline loss, high BC accuracy, or successfully written checkpoint is not treated as evidence of a stronger agent.
- Kaggle submission is outside this pipeline and remains disabled unless explicitly authorized again.

## Success criterion

Gold V3.1 succeeds only when a self-owned agent:

1. has zero runtime, illegal-action, and fallback failures;
2. beats its frozen parent under both seats;
3. improves against a diverse and recent public-opponent pool;
4. avoids unacceptable worst-matchup regression;
5. reproduces the gain under an independent 400-game seed table; and
6. later confirms the improvement through a valid Kaggle submission when separately authorized.

Until those conditions are met, G0 remains the champion and no gold-medal claim is made.
