<div align="center">

# 🏆 ZKAEDI PRIME Engine - Benchmark Leaderboard

**Real-time performance rankings** - Updated automatically via GitHub Actions

[![Auto Update](https://img.shields.io/badge/auto--update-daily-blue.svg)](.github/workflows/benchmark_leaderboard.yml)
[![Workflow](https://github.com/zkaedii/h-benchmark/workflows/Benchmark%20Leaderboard/badge.svg)](https://github.com/zkaedii/h-benchmark/actions/workflows/benchmark_leaderboard.yml)

</div>

*Last updated: 2026-01-03 00:13:37 UTC*

---

## 🥇 Top Performers

**Rankings by time per step (lower is better)**

| Rank | Configuration | Qubits | Time/Step (ms) | Entropy | Sparsity | Backend |
|:----:|:-------------|:------:|:--------------:|:-------:|:--------:|:-------:|
| 1 🥇 | `scalability_3q` | 3 | **0.056** | 0.0000 | 0.1250 | `dense` |
| 2 🥈 | `scalability_2q` | 2 | **0.063** | 0.0000 | 0.2500 | `dense` |
| 3 🥉 | `scalability_3q` | 3 | **0.064** | 0.0000 | 0.1250 | `dense` |
| 4  | `scalability_4q` | 4 | **0.065** | 0.0000 | 0.0625 | `dense` |
| 5  | `scalability_3q` | 3 | **0.066** | 0.0000 | 0.1250 | `dense` |
| 6  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 7  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 8  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 9  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 10  | `scalability_3q` | 3 | **0.067** | 0.0000 | 0.1250 | `dense` |
| 11  | `scalability_3q` | 3 | **0.068** | 0.0000 | 0.1250 | `dense` |
| 12  | `scalability_3q` | 3 | **0.069** | 0.0000 | 0.1250 | `dense` |
| 13  | `scalability_3q` | 3 | **0.069** | 0.0000 | 0.1250 | `dense` |
| 14  | `scalability_3q` | 3 | **0.069** | 0.0000 | 0.1250 | `dense` |
| 15  | `scalability_2q` | 2 | **0.070** | 0.0000 | 0.2500 | `dense` |
| 16  | `scalability_3q` | 3 | **0.070** | 0.0000 | 0.1250 | `dense` |
| 17  | `scalability_3q` | 3 | **0.071** | 0.0000 | 0.1250 | `dense` |
| 18  | `scalability_3q` | 3 | **0.071** | 0.0000 | 0.1250 | `dense` |
| 19  | `scalability_2q` | 2 | **0.071** | 0.0000 | 0.2500 | `dense` |
| 20  | `scalability_2q` | 2 | **0.073** | 0.0000 | 0.2500 | `dense` |

---

## 📊 Statistics

| Metric | Value |
|:-------|:------|
| **Fastest** | `scalability_3q` - **0.056 ms/step** |
| **Slowest** | `scalability_4q` - 0.103 ms/step |
| **Average** | 0.075 ms/step |
| **Median** | 0.074 ms/step |
| **Total Entries** | 66 |

### Backend Distribution

| Backend | Count | Percentage |
|:--------|:-----:|:----------:|
| `dense` | 66 | 100.0% |

---

## 📈 Performance by System Size

| Qubits | Best Time/Step (ms) | Average Time/Step (ms) | Entries |
|:------:|:-------------------:|:---------------------:|:-------:|
| 2 | **0.063** | 0.075 | 22 |
| 3 | **0.056** | 0.071 | 22 |
| 4 | **0.065** | 0.078 | 22 |

---

## 🔄 Recent Updates

| Timestamp | Entries Added |
|:----------|:--------------:|
| 2026-01-03 00:13:37 UTC | 3 |
| 2026-01-02 00:14:09 UTC | 3 |
| 2026-01-01 00:15:42 UTC | 3 |
| 2025-12-31 00:14:35 UTC | 3 |
| 2025-12-30 00:13:54 UTC | 3 |

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