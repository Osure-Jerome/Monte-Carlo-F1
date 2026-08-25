# Sprint 2 — Day 5 deliverables (100k statistics)

Per-run Monte Carlo exports for the Day-5 checkpoint (Monaco, 100,000 iterations
per strategy, paired by `sim_index`). These files are **generated artifacts**
and are therefore git-ignored; regenerate them with:

```bash
python -m f1strategist.cli --track Monaco --iterations 100000 --seed 42 \
    --parallel --csv data/sprint2/sprint2_100k_run1.csv
python -m f1strategist.cli --track Monaco --iterations 100000 --seed 43 \
    --parallel --csv data/sprint2/sprint2_100k_run2.csv
```

Verified results (recorded 2026-08-25):

| Batch | Win prob (A) | A mean | B mean | Stability |
| --- | ---: | ---: | ---: | --- |
| seed 42 (run 1) | 72.6 % | 6144.570 s | 6164.019 s | ±0.1 pp vs run 2 |
| seed 43 (run 2) | 72.5 % | 6144.530 s | 6163.920 s | NFR-4 target ±1 % ✔ |

Columns: `sim_index, strategy, total_time_s, pit_stop_count, sc_laps`.
Monaco mean `sc_laps` ≈ 43.6 vs stationary expectation `3p/(1+2p) × 78 = 43.9`.
