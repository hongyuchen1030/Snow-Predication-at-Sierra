# ForeSWE → T0 Factorized Spatiotemporal Transformer: Architecture Mapping

**Status: DESIGN DOCUMENT ONLY. No training, no implementation launched.** Per
instructions, this stops after the mapping for review.

## 0. Source

- **Paper:** Thapa, Krishu K., et al. *"ForeSWE: Forecasting Snow-Water
  Equivalent with an Uncertainty-Aware Attention Model."* Proceedings of the
  AAAI Conference on Artificial Intelligence, Vol. 40, No. 46, 2026.
- **Repository:** `https://github.com/Krishuthapa/SWE-Forecasting` (cloned
  locally at `thirdparties/SWE-Forecasting/`, confirmed via `git remote -v`).
- **Exact code inspected (the actual ForeSWE model, not the paper prose):**
  - `thirdparties/SWE-Forecasting/fore_swe/forecasting_att_daily_model.py` — the `SWETransformer` / `CrossVarTransformer` classes and full training loop, launched by `fore_swe/att_daily_run.sh` (README's designated "ForeSWE model" training script, no CLI hyperparameter overrides — the hardcoded defaults in this file **are** the actual settings used).
  - `thirdparties/SWE-Forecasting/fore_swe/custom_encoder_model.py` — the main encoder's `MultiHeadAttention` / `PositionWiseFeedForward` / `EncoderLayer`.
  - **Also inspected and explicitly ruled out as the primary reference:** `thirdparties/SWE-Forecasting/transformer_ar/model.py` — a much simpler standard `nn.Transformer` encoder-decoder ("transformer_ar" = a comparison baseline, not the model the README calls "ForeSWE"; different conda env name `transformer_env` vs. ForeSWE's `foreswe_att_env`).

---

## 1/2. What ForeSWE actually does

**Critical structural fact, stated up front: ForeSWE is a point/station-based
model, not a gridded spatial model.** It forecasts SWE at a fixed set of
~500 SNOTEL station locations (`test_indices` alone lists 100 station
indices up to 507; `train_loc_indices` adds more). There is no 2D image
grid, no patch embedding, no CNN stem anywhere in the ForeSWE code. Its
"tokens" for the main encoder are **station locations**, not spatial patches.
This matters a lot for adapting it to our gridded `[T,C,H,W]` task (see §4).

### A. Input representation
- Per-station driving variables: `encoder_input_dim=13` declared, but the
  code slices `encoder_inputs[:, :, 6:]` (columns 6+) as the actual
  "historical" per-variable data fed to the variable embeddings — this means
  only 7 of the 13 declared input columns are used as physical variables,
  with the first 6 columns split off separately as a "sp_tp" (space-time
  prompt) side-channel. **This 13 vs. 7 mismatch is unresolved/ambiguous in
  their own code** — flagged explicitly per instructions, not silently
  smoothed over.
- **Three parallel temporal scales**, each with its own window length fed to
  a per-scale `CrossVarTransformer` instance:
  - `historical_daily_att`: window=1 day (`batch_size=1` in their naming — a fixed source of confusion, this parameter is actually the temporal-window length, not a minibatch size).
  - `historical_monthly_att`: window=120 samples, step=7 days (≈2.3 years of weekly-strided history, despite being called "monthly").
  - `historical_yearly_att`: window=5 samples, step=365 days (5 years apart).
- Domain: CONUS SNOTEL network (point locations), not a fixed lat/lon grid.
- Tensor shape into one `CrossVarTransformer`: `[window_len, n_locations, n_vars]`.

### B. Spatial tokenization / embedding
- **No patch size — explicitly not applicable.** No convolutional embedding
  of any kind exists in the ForeSWE code.
- Embedding dimension: `model_dim=1024` per temporal-scale branch (constructor default, confirmed unmodified by the launch script).
- "Spatial tokens" = station locations (a point set, cardinality ≈500+, not derived from any patch/grid operation).

### C. Temporal representation
- Time is **not** attended over inside the main Transformer encoder at all.
  Instead, three independent temporal *scales* are each compressed by their
  own `CrossVarTransformer` (a single cross-attention operation, see D)
  **before** the main encoder even runs, then concatenated along the feature
  dimension: `torch.cat((daily, monthly, yearly), dim=2)` → `[1, n_locations, 3*model_dim]`.
- No explicit temporal positional encoding (sinusoidal or learned) over the
  T dimension exists — because there is no T dimension left by the time the
  main encoder runs; it has already been folded into the 3× feature
  concatenation.
- **Explicitly stated, per task instructions: ForeSWE does NOT resemble a
  factorized temporal→spatial attention design.** It has no "attend across T
  for each spatial token" block at all.

### D. "Spatial" attention (across station locations, in the per-scale branch AND the main encoder)
- **Per-scale branch (`CrossVarTransformer`):** not self-attention — a
  single `nn.MultiheadAttention` **cross**-attention call where the query is
  `sp_tp_embedding + prompt_embedding` (positional/metadata) and the
  key/value is the per-variable embedded input. This mixes across the
  **variable** dimension (7 variable-tokens, standard MHA, `nhead=16`), not
  across space.
- **Main encoder (`CustomEncoderLayer` / `MultiHeadAttention`):** full
  (non-windowed, non-local) self-attention across **all station locations
  simultaneously** — genuinely "global" attention over the point set, not
  neighborhood/local attention.
- **The one truly novel piece: geography-biased attention.** After the
  ordinary scaled-dot-product softmax, a **second** softmax is computed over
  a blend: `new_score = alpha*attn_scores + beta*(1/(1+distance)) +
  gamma*(1/(1+angularity))`, where `alpha,beta,gamma` are three learned
  scalars (shared per attention module) and `distance`/`angularity` are
  precomputed real-world pairwise geographic relationships between stations.
  This is an **additive geographic bias blended into full attention**, not a
  hard neighborhood mask/window — i.e. it is exactly the kind of mechanism
  your planned T1 "neighborhood-aware" modification would generalize, and is
  explicitly deferred (task §5).

### E. Transformer structure (main encoder)
- Layers: `enc_layers_count=8`.
- Attention heads: `nhead=16`.
- Hidden dim: `d_model = 3 * model_dim = 3072` (because the three temporal-scale branches are concatenated, not because 3072 was chosen directly).
- FFN dim: `d_ff=2048` (fixed default in `PositionWiseFeedForward`, **not** rescaled with `d_model=3072` — so the FFN expansion ratio is <1× here, an unusual choice, stated as found).
- Activation: **ReLU** inside the FFN; **GELU** everywhere else (variable-embedding MLP, output MLP) — inconsistent within their own code, noted as-is.
- LayerNorm placement: **Post-LN** — `x = LayerNorm(x + Dropout(sublayer(x)))` for both the attention and FFN sublayers.
- Residual structure: per-sublayer residuals (standard) **plus** one large outer residual around the *entire* 8-layer encoder stack: `final_output = encoder_stack_output + merged_input_x`.
- Dropout: `0.10` (encoder attention/FFN sublayers); a separate `0.05` dropout inside the small `sp_tp_prompt_embed` MLP and the output MLP.
- Attention dropout: not separately specified — `nn.MultiheadAttention`'s own default (0) inside `CrossVarTransformer`; the custom `MultiHeadAttention` class has no internal dropout at all (only the `EncoderLayer` wrapper applies dropout to the attention *output*).
- Stochastic depth: **not used** — not specified anywhere in the code.

### F. Positional information
- **Variable identity:** sinusoidal-*initialized* but **learnable**
  `nn.Parameter` (`variable_embeddings`), using the standard sin/cos formula
  applied to variable **index** (0..6), not to time or space. A ViT-style
  "initialize from a fixed formula, then fine-tune" trick.
- **Spatial position:** no explicit encoding — geography enters only through
  the distance/angularity attention bias (D), not an additive positional
  embedding.
- **Temporal position:** none in the main encoder (see C).
- **Station identity/metadata:** a separate `prompt_embedding`
  (`Linear(1536, model_dim)`) applied to a 1536-dim per-station "prompt"
  vector (contents not resolved from this file alone — likely a
  precomputed station-metadata or embedding vector; **ambiguous**, flagged).

### G. Output / readout
- No CLS token.
- No spatial pooling (all ~500 station tokens are carried through and
  produce independent per-station outputs — this is a per-location
  regression head, not a single global scalar).
- MLP funnel: `Linear(3072→1024) → GELU → Dropout(0.05) → Linear(1024→128) → GELU → Linear(128→8) → GELU`, producing an 8-dim **"final_representation"** per station-day.
- Final head: `Linear(8 → forecasting_window=10)` — **multi-horizon**: predicts the next 10 days of SWE simultaneously per station, not a single day.
- The 8-dim `final_representation` is saved separately and consumed by a
  **downstream Gaussian Process head** (`raw_gp/`, `daily_foreswe_gp_run.sh`)
  for the paper's "uncertainty-aware" calibrated forecast — **not applicable
  to us**; we have no uncertainty-quantification requirement in this task.

### H. Optimization
- Optimizer: **AdamW**.
- Learning rate: **5e-4** (`lr=0.0005`).
- Weight decay: **1e-4**.
- Scheduler: `StepLR(step_size=2, gamma=0.5)` — LR halved every 2 epochs.
- Warmup: **none**.
- "Batch size": effectively **1 gradient step per calendar day** in the
  training loop (`for index, input_index in enumerate(training_yr_indices)`,
  one `.backward()`/`.step()` per iteration) — there is no minibatching of
  independent samples; each single training step's "sequence length" is the
  full station count (~500+), not a batch dimension.
- Epochs: hard cap **`epoch_number <= 8`**.
- Early stopping: a "strike counter," patience **3**, tracking the
  **training loss itself** (`epoch_loss_sp`, the average per-epoch training
  MSE) — **not** a held-out validation loss inside this script. Test years
  are held out entirely separately for final evaluation only (README:
  train WY 1991–2014, test WY 2015–2019).
- Gradient clipping: **not used**.
- Mixed precision: **not used** (no autocast/GradScaler anywhere).
- Loss: `nn.MSELoss()` (the code names the variable `mae_error`, but it is
  literally MSE — a naming inconsistency in their source, stated as found).

### I. Parameter count
- **Not printed/logged anywhere in the code or README** — not available
  as-stated. My own computed estimate from the exact dimensions above
  (3 × `CrossVarTransformer` branches + 8× main encoder layers at
  `d_model=3072`, `nhead=16`, `d_ff=2048`, + output MLP) is **on the order
  of 190–210M parameters**, dominated by the `d_model=3072` main encoder's
  `Linear(3072,3072)` projections (4 per layer × 8 layers) and FFN. This is
  my calculation, not a ForeSWE-reported number — flagged as such.

---

## 3. ForeSWE → our T0 mapping

| ForeSWE component | ForeSWE setting | Our proposed T0 setting | Action | Reason |
|---|---|---|---|---|
| Input domain | ~500 irregular SNOTEL station points | 120×240 regular global grid | ADAPT | Fundamentally different input topology; a gridded field requires tokenization ForeSWE never does. |
| Spatial tokenization | none (point set) | Conv2d patch embedding, patch=8 → N=450 tokens (15×30) | NEW (not from ForeSWE) | No analogue exists in ForeSWE to reuse; patch size chosen so N≈450 lands close to ForeSWE's own demonstrated-feasible station count (~500), see §8. |
| Per-variable handling | separate `CrossVarTransformer` cross-attention over 7 variable-tokens | channels mixed jointly inside the patch-embedding conv | ADAPT (simplified) | Our channel count (12) is modest; folding channel-mixing into the embedding keeps T0 "the closest reasonable analogue to ordinary spatial attention" per task §5, avoiding ForeSWE's extra cross-variable-attention machinery built for a heterogeneous 13-input, GP-coupled pipeline we don't have. |
| Multi-scale temporal input | 3 parallel scales (daily/monthly/yearly), concatenated before the encoder | single scale, T=7 days | NOT APPLICABLE | Our task is fixed at T=7 daily steps by design; no multi-scale requirement. |
| Temporal attention | **none** (time compressed pre-encoder) | explicit temporal self-attention per spatial token across T=7, each block | NEW (task-mandated, not from ForeSWE) | Task §4 requires factorized temporal→spatial attention; ForeSWE has no analogous block to reuse. |
| Spatial attention | full self-attention over station tokens, **geography-bias blended into softmax** (distance+angularity) | full self-attention over the N=450 grid-patch tokens, **plain softmax(QKᵀ/√d)V, no bias term** | REUSE (mechanism) / DEFER (bias term) | Task §5 explicitly forbids the neighborhood-aware modification for T0; ForeSWE's own geography-bias attention is exactly the historical precedent for our *planned T1*, not T0. |
| Encoder sub-layer order | Self-Attn → FFN (LayerNorm-Post, per sublayer) | Temporal-Attn → Spatial-Attn → FFN (LayerNorm-Post, per sublayer) | ADAPT | Reuses ForeSWE's exact residual/Post-LN pattern per sublayer; adds the extra temporal sublayer required by the factorized-ST spec. |
| FFN | Linear→ReLU→Linear, `d_ff=2048` fixed (not scaled to d_model) | Linear→GELU→Linear, `d_ff=4×d_model` | ADAPT | Standardize on GELU (ForeSWE already uses GELU everywhere except the FFN — treat that as their inconsistency, not a deliberate choice); use the conventional 4× FFN ratio since ForeSWE's fixed 2048 was sized for their `d_model=3072`, not transferable to our much smaller d_model. |
| Dropout | 0.10 (encoder), 0.05 (small MLPs) | 0.10 (encoder blocks), 0.05 (embedding/head MLPs) | REUSE | Directly transferable, not compute-dependent. |
| Positional encoding (var/token identity) | sinusoidal-initialized, learnable `nn.Parameter`, applied to variable index | same mechanism, applied to (a) patch spatial index and (b) timestep index | REUSE (mechanism) | ForeSWE's "init from sin/cos, then learn" trick is a clean, directly reusable idea; only the *axis* it's applied to changes (space+time instead of variable identity), since we don't have ForeSWE's per-variable-token design (see row 3). |
| Geographic distance/angularity input | precomputed real station pairwise distance/angle | N/A (regular grid; adjacency is implicit in patch coordinates) | NOT APPLICABLE for T0 | Deferred to T1 exactly where it belongs — a regular grid's "distance" is already recoverable from the 2D positional embedding; the explicit bias term is the T1 feature, not T0. |
| Readout | per-station: MLP funnel 3072→1024→128→8, `Linear(8→10)` multi-day multi-station output | global average pool over N (and over T, using only the last timestep's tokens) → MLP funnel `d_model→64→1` | ADAPT | We need one Sierra-region scalar ΔSWE, not 10-day multi-station maps; reuse ForeSWE's "funnel down then small linear head" *style*, adapt the pooling and output dimensionality. |
| Downstream GP head | yes (uncertainty quantification) | none | NOT APPLICABLE | Out of scope for this pretraining-comparison stage; no uncertainty requirement stated for T0. |
| Optimizer / LR / WD | AdamW, 5e-4, 1e-4 | AdamW, 5e-4, 1e-4 | REUSE | Directly transferable and consistent with C0/C1's own AdamW choice already used in this project. |
| Scheduler | StepLR(2, 0.5) | ReduceLROnPlateau (as used for C0/C1) | ADAPT | Keeps the comparison against C0/C1 apples-to-apples (identical scheduler family across all three models, per task §6/§11 fairness requirement in the earlier C0/C1 spec), rather than introducing a fourth scheduler behavior only for T0. |
| Batch size / grad steps | 1 sample/step, full station set as "sequence" | standard minibatching (TBD, benchmark-driven like C0/C1) | ADAPT | ForeSWE's single-sample-per-step design is a side effect of their per-day station-set formulation, not a deliberate architectural choice worth preserving; C0/C1 already use standard minibatches and T0 should match that protocol for a fair comparison. |
| Epochs / early stopping | max 8 epochs, patience 3 **on training loss** | max 40 epochs, patience 6 **on validation loss** (as used for C0/C1) | ADAPT | Task §6 explicitly requires the same protocol as C0/C1 for comparability; ForeSWE's training-loss-based stopping is not a validation-based overfitting check at all and is not appropriate to reuse given our explicit overfitting-monitoring goal. |
| Mixed precision / grad clipping | neither used | both used (AMP + grad-clip, as for C0/C1) | ADAPT | Same fairness/comparability reasoning; also a Perlmutter A100 efficiency best-practice already established for C0/C1. |
| Loss | MSE | Huber (as used for C0/C1) | ADAPT | Task §6 explicitly requires identical loss across all models being compared. |

---

## 4. How closely does ForeSWE resemble factorized spatiotemporal attention?

**Not very closely — reported explicitly per instructions, before changing
anything.** ForeSWE has:
- No temporal-attention sublayer at all (time is compressed via 3 parallel
  pre-encoder branches + concatenation, not attended over inside the
  encoder).
- A single spatial (station-token) self-attention type, applied to the
  already-time-collapsed representation — there is exactly one attention
  "axis" active inside the main encoder stack, not two factorized axes.

So the factorized Temporal→Spatial→FFN block structure our task requires
(§4 of the task) is **not present in ForeSWE and must be newly built**; the
one piece of ForeSWE we *do* carry into that block is its spatial
self-attention mechanism (full, non-local MHA) and its Post-LN/residual/
dropout conventions — not its temporal-handling strategy, which doesn't
transfer to a factorized design at all.

---

## 5. Proposed T0 architecture, layer-by-layer

```
Input:            X ∈ R^[B, T=7, C=12, H=120, W=240]

1. Patch embedding (shared across all T, applied per-day):
     reshape  -> [B*T, 12, 120, 240]
     Conv2d(12 -> d_model=256, kernel=8, stride=8)   # patch=8
     reshape  -> [B, T=7, d_model=256, H'=15, W'=30]
     flatten spatial -> [B, T=7, N=450, d_model=256]

2. Add positional embeddings (sinusoidal-initialized, learnable -- ForeSWE's
   variable-embedding trick, reused on space+time axes instead):
     spatial_pos:  [1, 1, N=450, d_model]  (learned, sin/cos-initialized on patch index)
     temporal_pos: [1, T=7, 1, d_model]    (learned, sin/cos-initialized on t index)
     Z0 = patches + spatial_pos + temporal_pos        -> [B, 7, 450, 256]

3. Factorized ST Transformer block, repeated L=4 times:
     For block l = 1..4:
       a. Temporal attention (per spatial token, across T):
            reshape [B,7,450,256] -> [B*450, 7, 256]
            MHA(Q=K=V=x, nhead=8), Post-LN residual, dropout=0.10
            reshape back -> [B,7,450,256]
       b. Spatial attention (per timestep, across N):
            reshape [B,7,450,256] -> [B*7, 450, 256]
            MHA(Q=K=V=x, nhead=8), Post-LN residual, dropout=0.10
            (plain softmax(QK^T/sqrt(d))V -- NO ForeSWE-style geography bias; that is T1)
            reshape back -> [B,7,450,256]
       c. FFN:
            Linear(256 -> 1024) -> GELU -> Linear(1024 -> 256)
            Post-LN residual, dropout=0.10

4. Readout:
     take last timestep only (t=T-1, i.e. day "t", consistent with the
     7-day-history -> next-day-change task framing):  [B, 450, 256]
     global average pool over N=450                -> [B, 256]
     MLP funnel (ForeSWE style): Linear(256->64) -> GELU -> Linear(64->1)

Output:            delta_swe_hat ∈ R^[B]   (normalized ΔSWE_1d; report mm via
                    the same target_mu/target_sigma convention already used
                    for C0/C1)
```

## Tensor shapes through T0

| Stage | Shape |
|---|---|
| Input | `[B, 7, 12, 120, 240]` |
| After patch embed | `[B, 7, 450, 256]` |
| After temporal attention (per block) | `[B, 7, 450, 256]` |
| After spatial attention (per block) | `[B, 7, 450, 256]` |
| After FFN (per block) | `[B, 7, 450, 256]` (unchanged shape through all 4 blocks) |
| After last-timestep selection | `[B, 450, 256]` |
| After spatial pooling | `[B, 256]` |
| Output | `[B]` |

---

## 6. Parameter-count estimate (T0, as proposed above)

| Component | Params |
|---|---:|
| Patch embed conv (12→256, 8×8) | ≈196,864 |
| Spatial positional table (450×256) | 115,200 |
| Temporal positional table (7×256) | 1,792 |
| Per block: temporal MHA (4×256²) + spatial MHA (4×256²) + FFN (2×256×1024) + 3×LayerNorm | ≈1,053,000 |
| × 4 blocks | ≈4,213,000 |
| Readout MLP (256→64→1) | ≈16,500 |
| **Total** | **≈4.5M** |

This is **~14–17× larger than C0 (272,833) / C1 (321,473)**. That gap is
reported here for your review, not resolved unilaterally — Transformers are
typically far less parameter-efficient than CNNs/ConvLSTMs at small scale,
and this is already a substantial reduction from ForeSWE's own ~190–210M.
If you want closer parity with C0/C1 for this first baseline, dropping to
`d_model=128, L=3` (~1.1M params) is a straightforward, purely
hyperparameter-level change with no architectural impact — flagging as an
option rather than deciding it here.

## 7. Expected GPU-memory feasibility (one Perlmutter A100, 40GB)

For `T=7, H=120, W=240`, patch=8 → **N=450 spatial tokens**, **T×N = 3,150 total tokens**.

- **Temporal-attention complexity** (per spatial token, across T): `O(N · T² · d)` = `450 × 49 × 256` ≈ 5.6M multiply-adds per block per sample — trivial, T=7 is tiny.
- **Spatial-attention complexity** (per timestep, across N): `O(T · N² · d)` = `7 × 202,500 × 256` ≈ 363M multiply-adds per block per sample — this dominates, scaling **quadratically in N**, not in T.
- **Attention score matrix memory** (the actual feasibility bottleneck): `T × nhead × N² × 4 bytes` = `7 × 8 × 202,500 × 4 bytes` ≈ **45.4 MB per sample per block** for the spatial-attention scores alone. At a batch size of 32 and 4 blocks (not reusing memory across blocks during backward): `45.4MB × 32 × 4` ≈ **5.8 GB** — comfortably fits a 40GB A100 alongside activations, gradients, and optimizer state, with large headroom for AMP.
- **Sanity check against ForeSWE's own demonstrated scale:** ForeSWE's main encoder already runs full attention over ~500 station tokens (comparable order of magnitude to our N=450) for 8 layers at `d_model=3072` — our T0 uses a *smaller* d_model (256) at similar N, so this is a strictly lighter computational regime than what ForeSWE itself already trains successfully.
- **If a finer patch were used instead** (e.g. patch=4 → N=1,800, matching C1's ConvLSTM stem resolution): spatial-attention score memory becomes `7 × 8 × 1,800² × 4 bytes` ≈ **725 MB/sample/block** → at batch=32, 4 blocks: **≈93 GB**, exceeding one A100 without gradient checkpointing or a much smaller batch size.
- **Conclusion: patch=8 (N=450) is the recommended default** — it is not an arbitrary shrink; it is the minimum-necessary adaptation identified by this calculation, and it is independently justified by matching ForeSWE's own already-proven token-count regime (§8 instruction: don't shrink arbitrarily — this is the concrete calculation behind the choice).

---

## 8. Every deviation from ForeSWE, one sentence each

1. **Grid patch embedding added** — ForeSWE has no spatial tokenization at all (point-based); a gridded domain requires one.
2. **Cross-variable attention module dropped** — folded into the patch-embedding conv instead, to keep T0 a lean "ordinary spatial attention" baseline per task §5.
3. **Multi-scale (daily/monthly/yearly) branch structure dropped** — our task is fixed at a single T=7 daily scale by design.
4. **Explicit temporal-attention sublayer added** — required by the factorized-ST spec (task §4); ForeSWE has no analogous block.
5. **Geography-bias attention term (α/β/γ, distance/angularity) omitted from spatial attention** — that mechanism is the direct precedent for the *planned T1*, explicitly deferred for T0 per task §5.
6. **FFN activation changed ReLU→GELU** — standardizes with the GELU used everywhere else in ForeSWE's own code (their ReLU-in-FFN reads as an inconsistency, not a deliberate design choice).
7. **FFN expansion ratio changed to 4×d_model** — ForeSWE's fixed `d_ff=2048` was sized for their `d_model=3072` (a <1× ratio) and doesn't transfer to our much smaller d_model.
8. **Readout changed to global-average-pool + small MLP + single scalar** — our task needs one Sierra-region ΔSWE value, not ForeSWE's per-station 10-day multi-horizon map.
9. **No downstream Gaussian Process head** — no uncertainty-quantification requirement in this task.
10. **Scheduler changed StepLR→ReduceLROnPlateau, loss MSE→Huber, early-stopping training-loss→validation-loss, added AMP+grad-clip, standard minibatching instead of 1-sample/step** — all required to keep T0 directly comparable to the already-completed C0/C1 protocol (task §6/§7), not ForeSWE-specific choices.
11. **d_model reduced from 3072 (ForeSWE's concatenated 3-branch width) to 256** — ForeSWE's width was sized for a 3-branch-concatenation design we don't have and a much richer downstream (GP-coupled) pipeline; 256 keeps T0 in a tractable-but-still-meaningfully-Transformer regime (see §6 for the resulting ~4.5M-parameter estimate and the smaller-alternative option).
12. **Depth reduced from 8 layers to 4** — a compute/scale-budget choice for a first "clean baseline," reversible if underfitting is observed.

## 9. Remaining ambiguities in ForeSWE itself (not resolved, flagged as-is)

1. **`encoder_input_dim=13` vs. the actual 7 columns used** (`encoder_inputs[:, :, 6:]`) — internally inconsistent in their own code; unclear which is authoritative.
2. **Contents of the 6-dim `sp_tp` side-channel and the 1536-dim `prompt` vector** — not resolvable from this file alone (likely day-of-year/lat-lon/elevation and a station-embedding vector respectively, but not confirmed).
3. **ForeSWE's own parameter count** — never printed/logged; the ~190–210M figure above is my calculation, not their reported number.
4. **Whether `distances`/`angularities` are normalized/scaled before the `1/(1+x)` transform** — not shown in the excerpted code path used here.

---

**Stopping here for review, per instructions. No training or implementation launched.**
