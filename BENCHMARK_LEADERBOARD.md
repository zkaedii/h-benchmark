<div align="center">

# 🏆 ZKAEDI PRIME Engine - Benchmark Leaderboard

**Real-time performance rankings** - Updated automatically via GitHub Actions

[![Auto Update](https://img.shields.io/badge/auto--update-daily-blue.svg)](.github/workflows/benchmark_leaderboard.yml)
[![Workflow](https://github.com/zkaedii/h-benchmark/workflows/Benchmark%20Leaderboard/badge.svg)](https://github.com/zkaedii/h-benchmark/actions/workflows/benchmark_leaderboard.yml)

</div>

*Last updated: 2026-01-16 00:14:15 UTC*

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
| 7  | `scalability_2q` | 2 | **0.063** | 0.0000 | 0.2500 | `dense` |
| 8  | `scalability_3q` | 3 | **0.064** | 0.0000 | 0.1250 | `dense` |
| 9  | `scalability_3q` | 3 | **0.065** | 0.0000 | 0.1250 | `dense` |
| 10  | `scalability_4q` | 4 | **0.065** | 0.0000 | 0.0625 | `dense` |
| 11  | `scalability_4q` | 4 | **0.065** | 0.0000 | 0.0625 | `dense` |
| 12  | `scalability_3q` | 3 | **0.066** | 0.0000 | 0.1250 | `dense` |
| 13  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 14  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 15  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 16  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 17  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 18  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 19  | `scalability_3q` | 3 | **0.068** | 0.0000 | 0.1250 | `dense` |
| 20  | `scalability_3q` | 3 | **0.069** | 0.0000 | 0.1250 | `dense` |

---

## 📊 Statistics

| Metric | Value |
|:-------|:------|
| **Fastest** | `scalability_3q` - **0.052 ms/step** |
| **Slowest** | `scalability_4q` - 0.132 ms/step |
| **Average** | 0.076 ms/step |
| **Median** | 0.075 ms/step |
| **Total Entries** | 105 |

### Backend Distribution

| Backend | Count | Percentage |
|:--------|:-----:|:----------:|
| `dense` | 105 | 100.0% |

---

## 📈 Performance by System Size

| Qubits | Best Time/Step (ms) | Average Time/Step (ms) | Entries |
|:------:|:-------------------:|:---------------------:|:-------:|
| 2 | **0.059** | 0.076 | 35 |
| 3 | **0.052** | 0.072 | 35 |
| 4 | **0.058** | 0.079 | 35 |

---

## 🔄 Recent Updates

| Timestamp | Entries Added |
|:----------|:--------------:|
| 2026-01-16 00:14:15 UTC | 3 |
| 2026-01-15 00:14:07 UTC | 3 |
| 2026-01-14 00:14:35 UTC | 3 |
| 2026-01-13 00:12:01 UTC | 3 |
| 2026-01-12 00:15:08 UTC | 3 |

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