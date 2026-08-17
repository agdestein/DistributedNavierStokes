# Campaign readiness

What must happen in this solver before the second-generation DNS campaign
(`~/Projects/ProbabilisticClosure/data-campaign.md`; Snellius extension
draft `snellius-application.md`) can run. Like those files this one is
edited in place: items are checked off (with the commit or result that
closes them) as they land, and the S0 design below is rewritten if the
experiment changes.

Context: the application's cost/memory estimates are marked **[V0]** —
speculation pending measurement. The first action is therefore **S0, a
scaling experiment** (below), pulled ahead of the V0 physics twin: it
produces every number the application needs without waiting on any
physics decision.

## S0 — scaling and footprint experiment (**done 2026-08-14**, results below)

One cheap job (measured: ~20 min on a 4-H100 allocation, ≈ 300 SBU) that
replaces the application's guessed anchors with measurements. Re-run
after any transport/memory change: `sbatch examples/snellius/scaling.sh`
(dummy pipeline check: `sbatch --time=00:20:00 examples/snellius/scaling.sh dummy`).

### What it measures, and what each number is for

1. **Per-step wall time vs n and GPU count** — the SBU-per-simulated-
   time-unit pricing for R1–R3, and a check of the assumed ~N⁴ cost
   scaling (log–log slope over the size scan).
2. **Coarse-resolution plateau** — very small grids down to 48³, where
   per-step time should flatten into a fixed overhead floor (kernel
   launches, MPI latency, host-staging constants). Locates where the
   GPUs stop being utilized, bounds the per-step overhead that large
   runs also pay, and is the relevant number for LES-scale usage.
3. **Strong scaling at 810³ across 1/2/4 GPUs, plus procgrid variants**
   ((2,2) vs (1,4) vs (4,1) at 4 ranks) — the parallel efficiency
   honesty check and the procgrid choice for production.
4. **Per-rank GPU memory + the OOM boundary** — measured footprint at
   every config, plus deliberate probes at the ledger's marginal points:
   810³ on 1 GPU, 1080³ on 2, 1296³ on 4 (expected-fail allowed; each
   config is its own srun, so a probe OOM kills only its own row). This
   settles data-campaign §8's open decision on R1/R3's exact N.
5. **Communication share** — a transform-only microbenchmark
   (`spec_to_phys!`/`phys_to_spec!` round trips, timed like a step) next
   to the full step time, separating transpose+FFT cost from local
   kernels without instrumenting the solver. This is the number that
   decides how urgent the UCX-device-path/NCCL work is: everything runs
   `DNS_MPIBUF=host` (the only working mode on Snellius), so the comm
   share bounds what a working device path could win back.
6. **Snapshot I/O bandwidth** — one `spectral_save` timed at the largest
   sizes (3.8 GB at 810³, 12.3 GB at 1200³), on scratch and on project
   space: prices checkpoint cadence and the campaign's snapshot budgets.
7. **Optional riders** (cheap, same job): one `sfswriter` sample at 810³
   / nles = 128 (filtering was part of round one's 0.27 GPU·h/tu anchor,
   so pricing needs it); a small-n `mpibuf = :device` smoke to re-test
   whether the UCX stack still crashes on device buffers.

### Protocol

- **One config = one fresh `srun` Julia process** (clean device memory;
  compile time re-measured per config, which is itself a number worth
  having for job-time budgeting).
- **Warm-up then time** (compilation excluded from runtime): synthetic
  field from `spectral_randomfield!` (k⁻⁵ᐟ³-type profile, seed fixed),
  run 2 throw-away steps (the first carries compilation; the second
  confirms steady state), then `CUDA.synchronize()` + `MPI.Barrier` and
  time each of the next `nsteps` steps individually (barrier+sync per
  step). Report min / median / mean per-step time — min approximates the
  noise-free step; mean prices the run. Record separately: wall time of
  step 1 (compile+step) and of the whole process (job budgeting).
- `nsteps` scaled so the timed window is ≥ ~2 s: 50 for n ≤ 192, 20 for
  mid sizes, 10 at the largest.
- **Fixed Δt** for the timed steps (per-step work does not depend on the
  dt value, and it keeps configs identical); the CFL-proposed dt at the
  synthetic field is *recorded* per config for the steps-per-time-unit
  conversion, cross-anchored against round one's measured 0.27 GPU·h/tu
  at 810³ (the synthetic field's umax differs from the stationary
  state's, so the conversion carries that caveat explicitly).
- **Memory**: per-rank device memory after the timed steps
  (`JULIA_CUDA_MEMORY_POOL=none` is already the cluster setting, so
  used-memory is meaningful), gathered to rank 0, max reported.
- **Config matrix** (all n divisible by 3, FFT-smooth):
  - plateau, 1 and 4 ranks: n = 48, 96, 192
  - size scan, 1 GPU: n = 384, 576, 810 (810 is the OOM probe)
  - size scan, 2 GPUs: n = 810, 972, 1080 (1080 is the probe)
  - size scan, 4 GPUs: n = 810, 1080, 1200, 1296 (1296 is the probe)
  - procgrid, 4 ranks at 810³: (2,2), (1,4), (4,1)
- **Output**: one machine-readable row per config (TOML or CSV: n,
  ranks, procgrid, mpibuf, per-step min/median/mean, transform-only
  time, compile wall, dt_CFL, max rank memory, save seconds + GB/s,
  OOM/crash flag), collected by the job script; the summary table is
  committed back into this file, and the measured values replace the
  **[V0]** slots in `snellius-application.md` (SBU/tu at 810³ and 1200³,
  per-GPU memory margins, the scaling exponent — and fix its
  "slab-decomposed" wording: the solver is pencil-decomposed).

### Results (2026-08-14, jobs 25618452 + 25618693, ≈ 300 SBU total)

All host-staged MPI buffers (`DNS_MPIBUF=host`) except the device smoke.
Median per-step / transform-round-trip seconds, max per-GPU memory:

| n | GPUs | procgrid | step (s) | round trip (s) | GB/GPU |
|---|---|---|---|---|---|
| 48 | 1 | (1,1) | 0.0033 | 0.0006 | 0.6 |
| 96 | 1 | (1,1) | 0.0050 | 0.0009 | 0.8 |
| 192 | 1 | (1,1) | 0.0157 | 0.0028 | 2.3 |
| 384 | 1 | (1,1) | 0.116 | 0.021 | 13.9 |
| 576 | 1 | (1,1) | 0.455 | 0.086 | 45.5 |
| 810 | 1 | (1,1) | **OOM** | | > 94 |
| 48 | 4 | (2,2) | 0.0065 | 0.0013 | 0.6 |
| 96 | 4 | (2,2) | 0.0150 | 0.0033 | 0.7 |
| 192 | 4 | (2,2) | 0.0986 | 0.0215 | 1.1 |
| 810 | 2 | (2,1) | 4.93 | 1.08 | 69.6 |
| 810 | 2 | (1,2) | 4.19 | 0.90 | 68.2 |
| 972, 1080 | 2 | | **OOM** | | |
| 810 | 4 | (2,2) | 5.64 | 1.27 | 36.5 |
| 810 | 4 | (4,1) | 3.31 | 0.72 | 35.2 |
| 810 | 4 | (1,4) | **2.28** | 0.489 | 34.5 |
| 1080 | 4 | (2,2) | 13.3 | 2.82 | 85.5 |
| 1080 | 4 | (1,4) | **5.53** | 1.19 | 80.7 |
| 1200, 1296 | 4 | (2,2) | **OOM** | | |
| 192 | 4 | device path | **crash** (UCX, unchanged) | | |

Riders: `spectral_save` at 810³ wrote 3.55 GB at 1.3 GB/s to
scratch-shared (2.7 s — checkpointing is negligible); one SFS sample
(nles = 128, one Δ) 2.2 s; compile 2.6–16 s per process; `dt_cfl`
bit-identical across all rank counts and grids (invariance fix visible in
the data).

**NCCL transport rerun (2026-08-14, job 25619132, after the `:nccl`
transpose transport landed; correctness gated by the 4-vs-1 validation at
2.1e-16, bit-identical to the host-staged result):**

| n | GPUs | procgrid | step (s) | vs host | round trip (s) | GB/GPU |
|---|---|---|---|---|---|---|
| 48 | 4 | (2,2) | 0.0038 | 1.7× | 0.0007 | 1.3 |
| 96 | 4 | (2,2) | 0.0045 | 3.3× | 0.0009 | 1.3 |
| 192 | 4 | (2,2) | 0.0095 | 10.4× | 0.0019 | 1.7 |
| 810 | 2 | (1,2) | 0.870 | 4.8× | 0.172 | 68.7 |
| 810 | 4 | (2,2) | 0.523 | 10.8× | 0.106 | 37.1 |
| 810 | 4 | (4,1) | 0.438 | 7.6× | 0.087 | 36.3 |
| 810 | 4 | (1,4) | **0.383** | 5.9× | 0.074 | 35.6 |
| 1080 | 4 | (2,2) | 1.339 | 9.9× | 0.267 | 86.1 |
| 1080 | 4 | (1,4) | **1.102** | 5.0× | 0.215 | 81.9 |

NCCL findings: **steps are 5–11× faster than host-staged**. 810³ on 4
GPUs runs at 4.6× the extrapolated single-GPU step — essentially ideal
strong scaling (2→4 GPUs: 2.3×) — and the communication share of a
transform round trip drops from ~88% to ~20% (0.074 s round trip vs
≈ 0.06 s local FFT work). Procgrid (1,p) still wins, now by 1.37× rather
than 2.5×. NCCL costs ~1 GB/GPU extra (1080³ (2,2) at 86.1 GB still
fits). Wall-clock: 810³ is now 2.6× faster than round one's single-GPU
solver; 1080³ runs at 1.10 s/step. **Updated pricing** (same synthetic-dt
caveat): 810³ ≈ 0.41 GPU·h ≈ 80 SBU per time unit (1.5× round one's
per-tu rate, on 4× the hardware); 1080³ ≈ 1.6 GPU·h ≈ 300 SBU/tu; 810³
on 2 GPUs (R2 shape) ≈ 0.47 GPU·h ≈ 90 SBU/tu. These are the numbers the
application can carry.

**Memory-reduction rerun (2026-08-14, job 25629397, after freeing the
hidden CUDA.jl real-FFT plan buffers and sharing the transpose pair;
validated 4-vs-1 at 2.1e-16, bit-identical again):**

| n | GPUs | procgrid | step (s) | GB/GPU (before → after) |
|---|---|---|---|---|
| 810 | 4 | (1,4) | 0.375 | 35.6 → 28.3 |
| 972 | 2 | (1,2) | 1.354 | OOM → **92.7 (fits, tight)** |
| 1080 | 4 | (1,4) | 1.082 | 81.9 → 64.6 |
| 1200 | 4 | (1,4) | **2.022** | OOM → **87.9 (fits)** |
| 810 | 1 | | | still OOM |
| 1080 | 2 | (1,2) | | still OOM |
| 1296 | 4 | (1,4) | | still OOM |

1200³ pricing: dt_cfl 6.83e-4 → ≈ 3.3 GPU·h ≈ 630 SBU per time unit.
Snapshot save at 1200³: 11.5 GB in 17 s (scratch-shared bandwidth
varies 0.5-1.3 GB/s across jobs; still negligible against a law-mode
cadence).

**Production re-anchor (2026-08-16, from the completed R1 run):** all
SBU/tu prices above used the pre-a011e41 CFL-0.35 Δt heuristic. The
RK3-stability-boundary selector (a011e41, landed mid-R1 at the chunk-3
restart) takes **1.81× larger steps** on the same trajectory (mean Δt
6.84e-4 → 1.236e-3 at 972³, measured across the restart in R1's
stats.csv). Measured production price at 972³ on 2 GPUs: **0.61
GPU·h ≈ 118 SBU per time unit** (3.26 tu/h wall, incl. per-chunk
restart overhead) — matching the S0 step time × new Δt to ~1%. Scaled
S0 prices under the new selector: 810³/2 ≈ 50 SBU/tu, 810³/4 ≈ 44,
1080³/4 ≈ 165, 1200³/4 ≈ 350. These (and R1's actuals) are the numbers
now carried in data-campaign.md §7 / snellius-application.md.

**Findings:**

1. **Transforms are ~95% of a step, and staging is ~88% of a
   transform.** A step is 4.5 round-trip equivalents (3 stages × 1
   backward + 2 forward): 4.5 × 0.489 ≈ 2.20 of the 2.28 s step at 810³
   (1,4). Comparing the round trip against the single-GPU per-point FFT
   rate, transpose+host-staging is ≈ 88% of it. Consequence: 810³ on 4
   GPUs (2.28 s) is *slower* than the extrapolated single-GPU step
   (≈ 1.8 s from the 576³ point at the measured ~N³·⁴ per-step slope).
   Strong scaling *between* multi-GPU points is fine (2→4 GPUs: 4.19 →
   2.28 s, 92% efficiency) — the loss is the flat staging tax. The
   device path still crashes (UCX unchanged), so **NCCL is the measured
   top priority**, with roughly 2–3× wall-clock at stake.
2. **Memory is ≈ 2.2× the design ledger.** Fitted per-GPU model:
   ≈ 3.5 GB constant + ≈ 31 field-equivalents/ranks (ledger: ~14).
   Prime suspect: cuFFT plan work areas (six large plans, each O(data)).
   Consequences: 810³ does **not** fit one H100 (R2 as "1 GPU each"
   needs 2 GPUs, a leaner footprint, or SymmetryCode); 4 GPUs top out
   near 1080³–1130³, so **R3 at 1200³ is currently infeasible** —
   B_w = 1 batching plus FFT-workspace sharing lands ≈ 89 GB/GPU
   (marginal), or R3 moves to 1080³.
3. **Procgrid (1,p) beats squarest by 2–2.5×** ((1,4): 2.28 s vs (2,2):
   5.64 s at 810³; same at 1080³ and at 2 ranks). The spectral default
   (`squarest`) is the wrong choice at these rank counts; use explicit
   `procgrid = (1, p)` until the default is changed.
4. **Coarse plateau**: ≈ 3.3 ms/step floor on 1 GPU, ≈ 6.5 ms on 4
   ranks — irrelevant at DNS sizes, relevant if the solver is ever used
   at LES scale.
5. **Pricing at the current host-staged path** (dt from the synthetic
   field, caveat as designed): 810³ ≈ 2.5 GPU·h ≈ 470 SBU per time
   unit — **9× the round-one anchor** (0.27 GPU·h/tu single-GPU);
   1080³ ≈ 7.9 GPU·h ≈ 1500 SBU/tu. These numbers must not go into the
   application as-is: fix the transport first, then re-run S0 (the
   matrix is one cheap 20-min job).

## Readiness gaps, in priority order

Mapping: campaign items V0/R1/R2/R3, bursts, controls
(data-campaign.md §2–§6).

- [x] **1. NCCL transport** — done 2026-08-14 (commit 83b9f65 + job
  scripts). `mpibuf = :nccl` sends the packed transpose buffers over
  NCCL grouped send/recv (package extension, `using CUDA, NCCL`);
  correctness validated 4-vs-1 at 2.1e-16; S0 rerun shows 5–11× faster
  steps and near-ideal strong scaling (results above). Cluster
  specifics that were needed: the NCCL_jll stub (local-toolkit CUDA
  refuses JLL artifacts; `examples/snellius/NCCL_jll`, dev'd
  machine-locally) and job-level GPU allocation (`--gpus=4`, never
  `--gpus-per-task=1` — NCCL P2P needs all devices visible per rank).
  Production scripts default to NCCL now. Not done: the FV solver's
  transposes/halos still go over MPI only.
- [x] **2. Memory footprint reduction** — done 2026-08-14 (job
  25629397). The extra memory was not cuFFT work areas (those are zero
  for these plans) but hidden field-sized buffers inside CUDA.jl's
  out-of-place real-FFT plan objects (rfft: `ldiv!`-only, never used;
  brfft: input protection for a scratch buffer) plus double-allocated
  transpose buffers. Freed via the CUDA package extension (which also
  executes brfft directly — skips a full-field copy per backward
  transform, ~2% faster steps) and one send/recv pair shared by both
  transpose stages. 31 → ~25 equivalents; ledger rewritten
  (DESIGN_SPECTRAL §9). **Measured capacities: 1200³ on 4 H100s fits
  (87.9 GB/GPU, 2.02 s/step — R3 as specced is feasible, ≈ 630
  SBU/tu); 972³ on 2 fits (92.7 GB, 1-2 GB headroom); 810³ on 1,
  1080³ on 2, 1296³ on 4 still do not.** Remaining lever if those are
  ever needed: B_w = 1 product batching (−4.5 equivalents, unbuilt).
  R2 runs on 2 GPUs per realization (0.87 s/step at 810³).
- [x] **2b. Spectral procgrid default** — done 2026-08-14: the spectral
  default is now `(1, nranks)` slabs (measured fastest at 2-4 GPUs under
  both transports; the docstring says to pass an explicit grid beyond
  the measured range). Flipping the default also exposed a latent bug
  the suite then caught: the host-staged path copied whole (possibly
  reused, oversized) device buffers into exactly-sized host mirrors —
  now prefix copies. FV `setup` keeps `squarest`.
- [x] **3. Filter bank generalization** — done 2026-08-14. A bank cell
  is `(; M, kernel, Δfac)` (+ optional `Δη` label): kernels
  `:gaussian | :cutoff | :tophat | :helmholtz` sharing the low-k
  carve-out composite; multiple M per run via one gather at the largest
  M with rank-0 cube extraction (identical to a direct gather);
  `etacells(...)` pins widths in Δ/η at measured η with the
  choose-M-per-column window rule; `outtype = M -> M ≥ 256 ? Float32 :
  Float64` for bulky cells. Naming: a single-M all-Gaussian bank keeps
  SymmetryCode's flat `delta=<Δf>/` (loader-compatible, plus new
  `kernel`/`M`/`delta_eta` metadata keys); any other bank nests
  `filter=<kernel>/M=<M>/delta=<Δf>/`, so the downstream loader pointed
  at one (kernel, M) directory sees the flat layout it expects.
  Validated against a fresh FFTW oracle (all four kernels × two Ms, all
  processor grids, offline and legacy paths).
- [x] **4. Offline filtering driver** — done 2026-08-14. `sfs_offline`
  runs the same collector as the in-situ writer over stored snapshots
  (multi-GPU — σ̂ needs the full DNS-grid nonlinearity);
  `examples/spectral_filterbank.jl` self-configures from the snapshot
  sidecars and uses the leanest alias-free transform grid (n = 3·kcut,
  independent of the run's grid). The kinematic null is
  `spectral_phaserandomize!` (counter-based, Hermitian-safe: preserves
  every modal energy and incompressibility exactly, destroys phase
  correlations, decomposition-invariant — tested) via
  `DNS_PHASESEED`/`phaseseed`. GPU end-to-end verified (4 ranks:
  snapshots → bank → null bank).
- [x] **5. Run-record hygiene** — done 2026-08-14: `statswriter(; file,
  nupdate)` appends a CSV row of every K41 statistic per interval (rank
  0, append mode so restarts continue the series — the ε/L_int
  stationarity drift record; wired into `examples/spectral_hit.jl`),
  and `snapshotsaver` now stamps `eta`, `t_int`, `e`, `diss` measured
  at save time into each sidecar's meta (so snapshot spacing in t_int
  and Δ/η pinning are self-documenting; disable with `stats = false`).
- [x] **6. V0 twin-comparison design** — decided + tooling built
  2026-08-14: **exact twin via state import**, not statistical.
  `spectral_from_rfft!` maps a SymmetryCode full-rfft `(; x, y, z)`
  state (shared û = F[u]/n³ normalization) into the truncated state;
  `examples/symmetrycode_import.jl` converts a warmed-DNS JLD2 into a
  snapshot (serial, once), which any rank count then restarts from —
  so V0 compares both solvers evolving the *same* field, with the
  statistical spectra/statistics check as a byproduct of `statswriter`
  + `dns_meta`. Round-trip import is tested exact.
  **Local twin dry run passed 2026-08-14** (64³, ν = 5e-3, Re_λ ≈ 52,
  kmax·η ≈ 1.7; `examples/symmetrycode_twin_dns.jl` +
  `symmetrycode_twin_check.jl`): SymmetryCode warm-up → import → both
  solvers replay the same forced window at matched fixed Δt. Statistics
  of the imported field match to 1e-14; the final-field difference is
  purely the viscous-treatment difference (integrating factor vs
  explicit) and converges at exactly 3rd order: rel L2 5.8e-6 → 7.3e-7 →
  9.1e-8 for Δt = 1/64 → 1/128 → 1/256 over T = 1. The V0 twin recipe is
  verified end-to-end; on Snellius only scale changes.

Not needed for this campaign: the 2D slice writer (nothing in the spec
asks for it); Lundgren/linear forcing (optional control tier only —
implement only if that run is funded).

Already in place (validated): Float64 truncated-coefficient raw store =
exactly P1's format, rank-count-independent with machine-precision
restart; checkpoint/SLURM-resubmit for multi-day runs; law/pairing/burst
cadences via `tstops`; decomposition invariance incl. adaptive CFL
(Snellius 4×H100-vs-1, 2e-16); all campaign grid sizes divisible by 3 and
FFT-smooth (FFT lengths are always n; the prime m = 2n/3+1 is never
transformed); η recorded as `l_kol` in `statistics_dns`.

## Run scripts

- **R1** (`examples/spectral_r1.jl` + `snellius/spectral_r1.sh`, built
  2026-08-14): the ν = 1e-4 resolution-check DNS. Defaults to 972³ on
  2 H100s (the measured tight fit; `sbatch --ntasks=4 --gpus=4 …
  spectral_r1.sh 1080` for the roomier variant at ~1.5× the SBU/tu).
  Three restartable phases: warm-up to t = 25 (≈ 3 nominal t_int, drift
  in stats.csv), a sampling schedule fixed from the warm-up-end
  *measured* t_int (law: every t_int/2 over 10 t_int; test-style: 40
  over the first t_int) persisted to schedule.toml, then production with
  raw-f64 snapshots (P1 — the bank runs offline afterwards via
  `spectral_filterbank.jl`). `sbatch spectral_r1.sh dummy` runs the
  whole pipeline tiny — do that before every real submission.
  Building it exposed and fixed a production-blocking restart bug:
  `snapshotsaver` now takes `tstart` (the checkpoint's time) so restarts
  skip already-saved entries instead of rewriting every earlier snapshot
  index with the restart-time field. Verified locally end-to-end at 256³
  incl. a mid-production stop/resume: earlier snapshots byte-untouched,
  schedule reused, stats.csv single-headered, exact completion.

  **R1 ran to completion 2026-08-14→16** (jobs 25639420 → 25685620 →
  25702576 → 25714107, `sbatch spectral_r1.sh 972
  /projects/prjs1757/dns2/r1`): 4 auto-resubmitted chunks, 45.0 h wall
  on 2 H100s = **17.3k SBU** (vs the 11k estimate: t_int measured 8.74
  against ~7 assumed, and the first 23.5 h ran the old CFL-0.35
  selector — the a011e41 RK3-boundary selector landed at the chunk-3
  restart and took 1.81× larger steps; the same run costs ≈ 13k / 34 h
  today). Warm-up to t = 25 (2.9 t_int), schedule fixed at t_int =
  8.7379, production to t = 112.379: **59 raw f64 snapshots, 0.40 TB
  incl. checkpoints, on disk**. Physics: Re_λ 389–432, k_max·η
  1.29–1.38 over the sampled window (the resolution target ≥ 1.3, met
  in the mean at 1.33, grazed at dissipation peaks), E drifting 0.200 →
  0.221 over the 10-t_int window (shell clamp pins shells 1–2 only) —
  the drift record stats.csv was built for. Every restart invariant
  held in production (snapshots untouched across chunk boundaries,
  single-header stats, schedule reused, exact landing on the last
  scheduled time). The filter bank / twin analysis on these snapshots
  is post-processing via `spectral_filterbank.jl`.

  **R1 filter bank generated 2026-08-17** (jobs 25747606 + 25747607
  after dummy validation 25746483; driver env-configured and
  `examples/snellius/spectral_filterbank.sh` added in 08a9ec8, verified
  end-to-end locally first). Two passes, both 2 H100s, ≈ 250 SBU total,
  η pinned to the all-sidecar mean 0.0027362 so matched-Δ/η columns
  match exactly across M:
  - `full` → `/projects/prjs1757/dns2/r1/bank` (93 GB, 18 min): all 59
    snapshots × 11 cells — legacy Gaussian Δ/h {2.5, 3.5, 5} at M = 128
    (round one's test columns, the resolution-check comparison) plus
    Δ/η {40, 60} × {gaussian, cutoff, tophat, helmholtz} at M = 128
    (Δfac 2.23, 3.344).
  - `m256` → `/projects/prjs1757/dns2/r1/bank256` (130 GB, 20 min): the
    19 decorrelated law-mode snapshots × 12 cells — Δ/η {18, 27, 40} ×
    the four kernels at M = 256 (Δfac 2.007, 3.01, 4.459), Float32.
  The window rule dropped every M = 64 combo at this Re (widest column
  Δ/η = 60 is Δ/h = 1.67 there), as the campaign design predicts for
  the high-Re row.
