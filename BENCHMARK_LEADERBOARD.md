<div align="center">

# 🏆 ZKAEDI PRIME Engine - Benchmark Leaderboard

**Real-time performance rankings** - Updated automatically via GitHub Actions

[![Auto Update](https://img.shields.io/badge/auto--update-daily-blue.svg)](.github/workflows/benchmark_leaderboard.yml)
[![Workflow](https://github.com/zkaedii/h-benchmark/workflows/Benchmark%20Leaderboard/badge.svg)](https://github.com/zkaedii/h-benchmark/actions/workflows/benchmark_leaderboard.yml)

</div>

*Last updated: 2026-02-06 01:26:27 UTC*

---

## 🥇 Top Performers

**Rankings by time per step (lower is better)**

| Rank | Configuration | Qubits | Time/Step (ms) | Entropy | Sparsity | Backend |
|:----:|:-------------|:------:|:--------------:|:-------:|:--------:|:-------:|
| 1 🥇 | `scalability_3q` | 3 | **0.052** | 0.0000 | 0.1250 | `dense` |
| 2 🥈 | `scalability_3q` | 3 | **0.056** | 0.0000 | 0.1250 | `dense` |
| 3 🥉 | `scalability_4q` | 4 | **0.058** | 0.0000 | 0.0625 | `dense` |
| 4  | `scalability_2q` | 2 | **0.059** | 0.0000 | 0.2500 | `dense` |
| 5  | `scalability_2q` | 2 | **0.060** | 0.0000 | 0.2500 | `dense` |
| 6  | `scalability_3q` | 3 | **0.061** | 0.0000 | 0.1250 | `dense` |
| 7  | `scalability_3q` | 3 | **0.061** | 0.0000 | 0.1250 | `dense` |
| 8  | `scalability_2q` | 2 | **0.062** | 0.0000 | 0.2500 | `dense` |
| 9  | `scalability_3q` | 3 | **0.063** | 0.0000 | 0.1250 | `dense` |
| 10  | `scalability_3q` | 3 | **0.063** | 0.0000 | 0.1250 | `dense` |
| 11  | `scalability_2q` | 2 | **0.063** | 0.0000 | 0.2500 | `dense` |
| 12  | `scalability_2q` | 2 | **0.063** | 0.0000 | 0.2500 | `dense` |
| 13  | `scalability_3q` | 3 | **0.064** | 0.0000 | 0.1250 | `dense` |
| 14  | `scalability_3q` | 3 | **0.064** | 0.0000 | 0.1250 | `dense` |
| 15  | `scalability_4q` | 4 | **0.064** | 0.0000 | 0.0625 | `dense` |
| 16  | `scalability_3q` | 3 | **0.064** | 0.0000 | 0.1250 | `dense` |
| 17  | `scalability_3q` | 3 | **0.065** | 0.0000 | 0.1250 | `dense` |
| 18  | `scalability_4q` | 4 | **0.065** | 0.0000 | 0.0625 | `dense` |
| 19  | `scalability_4q` | 4 | **0.065** | 0.0000 | 0.0625 | `dense` |
| 20  | `scalability_3q` | 3 | **0.065** | 0.0000 | 0.1250 | `dense` |

---

## 📊 Statistics

| Metric | Value |
|:-------|:------|
| **Fastest** | `scalability_3q` - **0.052 ms/step** |
| **Slowest** | `scalability_4q` - 0.132 ms/step |
| **Average** | 0.075 ms/step |
| **Median** | 0.074 ms/step |
| **Total Entries** | 168 |

### Backend Distribution

| Backend | Count | Percentage |
|:--------|:-----:|:----------:|
| `dense` | 168 | 100.0% |

---

## 📈 Performance by System Size

| Qubits | Best Time/Step (ms) | Average Time/Step (ms) | Entries |
|:------:|:-------------------:|:---------------------:|:-------:|
| 2 | **0.059** | 0.075 | 56 |
| 3 | **0.052** | 0.071 | 56 |
| 4 | **0.058** | 0.078 | 56 |

---

## 🔄 Recent Updates

| Timestamp | Entries Added |
|:----------|:--------------:|
| 2026-02-06 01:26:27 UTC | 3 |
| 2026-02-05 01:25:24 UTC | 3 |
| 2026-02-04 01:25:24 UTC | 3 |
| 2026-02-03 01:29:47 UTC | 3 |
| 2026-02-02 01:29:39 UTC | 3 |

---

## 📝 Notes

- ✅ Leaderboard is automatically updated via GitHub Actions (daily at 00:00 UTC)
- ✅ Rankings are based on time per step (lower is better)
- ✅ Only successful benchmarks are included
- ✅ Results are sorted by performance (fastest first)
- ✅ Historical data is preserved in `benchmark_history.json`

### 🔧 Manual Update

To manually update the leaderboard:

```bash
# Run benchmarks
python -m zkaedi_prime_engine.benchmark

# Generate leaderboard
python -m zkaedi_prime_engine.leaderboard
```

Or trigger the workflow manually:
- Go to [Actions](https://github.com/zkaedii/h-benchmark/actions)
- Select 'Benchmark Leaderboard' workflow
- Click 'Run workflow'

*This leaderboard is maintained automatically. For manual updates, see the benchmark suite.*