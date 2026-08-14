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

- [ ] **1. NCCL transport** (blocks pricing → blocks the application).
  S0 measured: transforms are ~95% of a step and host staging ~88% of a
  transform; 4 GPUs at 810³ are slower than one (hypothetical) GPU, and
  the UCX device path still crashes (retested 2026-08-14). The seam is
  reserved, not built. After it lands: re-run S0 (one cheap job) and
  only then write the application's compute numbers.
- [ ] **2. Memory footprint reduction** (settles R1/R3's exact N).
  S0 measured ≈ 31 field-equivalents + 3.5 GB/GPU against the ledger's
  ~14: 810³ OOMs on one H100, 1200³ OOMs on 4. To do: find where the
  extra ~17 equivalents live (cuFFT plan work areas suspected — measure,
  then share/free work areas), implement B_w = 1 product batching
  (−4.5 equivalents), update the ledger to match reality. Target:
  1200³ on 4 GPUs (≈ 89 GB/GPU projected — marginal); fallback is R3
  at 1080³ (fits today: 80.7 GB/GPU) and R2 on 2 GPUs per run.
- [ ] **2b. Spectral procgrid default**: (1,p) beats `squarest` 2–2.5×
  at 2 and 4 ranks (S0 finding 3). Either flip the spectral default to
  slabs-in-dim-2 or benchmark the crossover at higher rank counts
  first; meanwhile production scripts pass `procgrid = (1, p)`
  explicitly.
- [ ] **3. Filter bank generalization** (the campaign's "free axis",
  needed before R1/R2 data lands, not before S0/V0):
  - [ ] kernels beyond Gaussian: sharp cutoff, top-hat (spectral sinc,
    negative lobes intended), optional Helmholtz — small spectral
    kernels in `les.jl`, uniform low-k carve-out composite;
  - [ ] multiple coarse grids M ∈ {64, 128, 256} per run (generalize
    `lessampler`/`sfswriter` or instantiate one per M); Float32 output
    option for the M = 256 bank cells (raw states stay Float64);
  - [ ] widths pinned in Δ/η (currently Δ/h factors) using measured η;
  - [ ] directory/metadata naming for the (kernel, Δ, M) cell axis —
    extends the SymmetryCode `delta=<Δf>/` schema; decide deliberately,
    the downstream loader consumes it.
- [ ] **4. Offline filtering driver.** The bank must be regenerable from
  the raw store (P1), but filtering only runs in-situ today. Pieces
  exist (`spectral_load!` + gather + `sfs_sample!`); needed: a
  multi-GPU driver looping stored snapshots through the bank (σ̂ needs
  the full DNS-grid nonlinearity — not a serial job at 1200³). The
  kinematic-null pass (randomized phases, refilter) lives here too.
- [ ] **5. Run-record hygiene** (small): statistics time series (ε,
  L_int, t_int, η per interval) written to file by a processor for the
  stationarity drift check; η/t_int stamped into snapshot sidecars.
- [ ] **6. V0 twin-comparison design.** The counter-based IC cannot
  bit-reproduce SymmetryCode's RNG stream. Either compare statistically
  (spectra/statistics over a window) or hand both solvers the same
  initial field via the snapshot format. Decide before V0 is scripted.
  (`spectral_randomfield!` already takes the round-one k⁻⁵ᐟ³ profile via
  its `profile` kwarg; forcing clamp + shells 1–3 control exist.)

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
