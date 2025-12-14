<div align="center">

# 🏆 ZKAEDI PRIME Engine - Benchmark Leaderboard

**Real-time performance rankings** - Updated automatically via GitHub Actions

[![Auto Update](https://img.shields.io/badge/auto--update-daily-blue.svg)](.github/workflows/benchmark_leaderboard.yml)
[![Workflow](https://github.com/zkaedii/h-benchmark/workflows/Benchmark%20Leaderboard/badge.svg)](https://github.com/zkaedii/h-benchmark/actions/workflows/benchmark_leaderboard.yml)

</div>

*Last updated: 2025-12-14 16:09:36 UTC*

---

## 🥇 Top Performers

**Rankings by time per step (lower is better)**

| Rank | Configuration | Qubits | Time/Step (ms) | Entropy | Sparsity | Backend |
|:----:|:-------------|:------:|:--------------:|:-------:|:--------:|:-------:|
| 1 🥇 | `scalability_3q` | 3 | **0.069** | 0.0000 | 0.1250 | `dense` |
| 2 🥈 | `scalability_2q` | 2 | **0.074** | 0.0000 | 0.2500 | `dense` |
| 3 🥉 | `scalability_2q` | 2 | **0.078** | 0.0000 | 0.2500 | `dense` |
| 4  | `scalability_4q` | 4 | **0.081** | 0.0000 | 0.0625 | `dense` |
| 5  | `scalability_3q` | 3 | **0.098** | 0.0000 | 0.1250 | `dense` |
| 6  | `scalability_4q` | 4 | **0.103** | 0.0000 | 0.0625 | `dense` |

---

## 📊 Statistics

| Metric | Value |
|:-------|:------|
| **Fastest** | `scalability_3q` - **0.069 ms/step** |
| **Slowest** | `scalability_4q` - 0.103 ms/step |
| **Average** | 0.084 ms/step |
| **Median** | 0.079 ms/step |
| **Total Entries** | 6 |

### Backend Distribution

| Backend | Count | Percentage |
|:--------|:-----:|:----------:|
| `dense` | 6 | 100.0% |

---

## 📈 Performance by System Size

| Qubits | Best Time/Step (ms) | Average Time/Step (ms) | Entries |
|:------:|:-------------------:|:---------------------:|:-------:|
| 2 | **0.074** | 0.076 | 2 |
| 3 | **0.069** | 0.083 | 2 |
| 4 | **0.081** | 0.092 | 2 |

---

## 🔄 Recent Updates

| Timestamp | Entries Added |
|:----------|:--------------:|
| 2025-12-14 16:09:36 UTC | 3 |
| 2025-12-14 16:08:25 UTC | 3 |

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