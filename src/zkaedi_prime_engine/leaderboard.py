"""Benchmark Leaderboard Generator

Generates a real-time leaderboard of benchmark results for display on GitHub.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from .benchmark import ZKAEDIPrimeBenchmark, BenchmarkResult


@dataclass
class LeaderboardEntry:
    """Single leaderboard entry."""
    rank: int
    config_name: str
    num_qubits: int
    time_per_step_ms: float
    final_entropy: float
    final_sparsity: float
    backend: str
    timestamp: str
    success: bool


class BenchmarkLeaderboard:
    """Generate and maintain benchmark leaderboard."""
    
    def __init__(self, output_file: str = "BENCHMARK_LEADERBOARD.md"):
        """Initialize leaderboard generator.
        
        Args:
            output_file: Output markdown file path
        """
        self.output_file = Path(output_file)
        self.entries: List[LeaderboardEntry] = []
    
    def load_existing(self):
        """Load existing leaderboard entries from history file."""
        self._load_history()
    
    def _load_history(self):
        """Load historical entries from JSON if available."""
        history_file = Path("benchmark_history.json")
        if history_file.exists():
            try:
                with open(history_file) as f:
                    history = json.load(f)
                    # Merge with existing entries (avoid duplicates)
                    existing_configs = {(e.config_name, e.timestamp) for e in self.entries}
                    for entry_data in history.get('entries', []):
                        key = (entry_data.get('config_name'), entry_data.get('timestamp'))
                        if key not in existing_configs:
                            entry = LeaderboardEntry(**entry_data)
                            self.entries.append(entry)
                            existing_configs.add(key)
            except Exception as e:
                print(f"Warning: Could not load history: {e}")
    
    def save_history(self):
        """Save entries to history file."""
        history_file = Path("benchmark_history.json")
        history = {
            'entries': [asdict(e) for e in self.entries],
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def add_results(self, results: List[BenchmarkResult]):
        """Add benchmark results to leaderboard.
        
        Args:
            results: List of benchmark results
        """
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        for result in results:
            if result.success:
                entry = LeaderboardEntry(
                    rank=0,  # Will be set after sorting
                    config_name=result.config_name,
                    num_qubits=result.num_qubits,
                    time_per_step_ms=result.time_per_step * 1000,
                    final_entropy=result.final_entropy,
                    final_sparsity=result.final_sparsity,
                    backend=result.final_backend,
                    timestamp=timestamp,
                    success=True
                )
                self.entries.append(entry)
    
    def generate_leaderboard(self) -> str:
        """Generate leaderboard markdown content.
        
        Returns:
            Markdown string for leaderboard
        """
        if not self.entries:
            return self._empty_leaderboard()
        
        # Sort by time per step (fastest first)
        sorted_entries = sorted(self.entries, key=lambda x: x.time_per_step_ms)
        
        # Assign ranks
        for i, entry in enumerate(sorted_entries, 1):
            entry.rank = i
        
        # Generate markdown
        md = []
        md.append("<div align=\"center\">")
        md.append("")
        md.append("# 🏆 ZKAEDI PRIME Engine - Benchmark Leaderboard")
        md.append("")
        md.append("**Real-time performance rankings** - Updated automatically via GitHub Actions")
        md.append("")
        md.append("[![Auto Update](https://img.shields.io/badge/auto--update-daily-blue.svg)](.github/workflows/benchmark_leaderboard.yml)")
        md.append("[![Workflow](https://github.com/zkaedii/h-benchmark/workflows/Benchmark%20Leaderboard/badge.svg)](https://github.com/zkaedii/h-benchmark/actions/workflows/benchmark_leaderboard.yml)")
        md.append("")
        md.append("</div>")
        md.append("")
        md.append(f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        md.append("")
        md.append("---")
        md.append("")
        
        # Top performers section
        md.append("## 🥇 Top Performers")
        md.append("")
        md.append("**Rankings by time per step (lower is better)**")
        md.append("")
        md.append("| Rank | Configuration | Qubits | Time/Step (ms) | Entropy | Sparsity | Backend |")
        md.append("|:----:|:-------------|:------:|:--------------:|:-------:|:--------:|:-------:|")
        
        # Show top 20
        top_entries = sorted_entries[:20]
        for entry in top_entries:
            medal = ""
            if entry.rank == 1:
                medal = "🥇"
            elif entry.rank == 2:
                medal = "🥈"
            elif entry.rank == 3:
                medal = "🥉"
            
            md.append(
                f"| {entry.rank} {medal} | `{entry.config_name}` | {entry.num_qubits} | "
                f"**{entry.time_per_step_ms:.3f}** | {entry.final_entropy:.4f} | "
                f"{entry.final_sparsity:.4f} | `{entry.backend}` |"
            )
        
        md.append("")
        md.append("---")
        md.append("")
        
        # Statistics section
        md.append("## 📊 Statistics")
        md.append("")
        
        if sorted_entries:
            fastest = sorted_entries[0]
            slowest = sorted_entries[-1]
            avg_time = np.mean([e.time_per_step_ms for e in sorted_entries])
            median_time = np.median([e.time_per_step_ms for e in sorted_entries])
            
            md.append("| Metric | Value |")
            md.append("|:-------|:------|")
            md.append(f"| **Fastest** | `{fastest.config_name}` - **{fastest.time_per_step_ms:.3f} ms/step** |")
            md.append(f"| **Slowest** | `{slowest.config_name}` - {slowest.time_per_step_ms:.3f} ms/step |")
            md.append(f"| **Average** | {avg_time:.3f} ms/step |")
            md.append(f"| **Median** | {median_time:.3f} ms/step |")
            md.append(f"| **Total Entries** | {len(sorted_entries)} |")
            md.append("")
        
        # Backend distribution
        md.append("### Backend Distribution")
        md.append("")
        backend_counts = {}
        for entry in sorted_entries:
            backend_counts[entry.backend] = backend_counts.get(entry.backend, 0) + 1
        
        md.append("| Backend | Count | Percentage |")
        md.append("|:--------|:-----:|:----------:|")
        total = len(sorted_entries)
        for backend, count in sorted(backend_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            md.append(f"| `{backend}` | {count} | {percentage:.1f}% |")
        
        md.append("")
        md.append("---")
        md.append("")
        
        # System size performance
        md.append("## 📈 Performance by System Size")
        md.append("")
        md.append("| Qubits | Best Time/Step (ms) | Average Time/Step (ms) | Entries |")
        md.append("|:------:|:-------------------:|:---------------------:|:-------:|")
        
        by_qubits = {}
        for entry in sorted_entries:
            if entry.num_qubits not in by_qubits:
                by_qubits[entry.num_qubits] = []
            by_qubits[entry.num_qubits].append(entry.time_per_step_ms)
        
        for qubits in sorted(by_qubits.keys()):
            times = by_qubits[qubits]
            best = min(times)
            avg = np.mean(times)
            md.append(f"| {qubits} | **{best:.3f}** | {avg:.3f} | {len(times)} |")
        
        md.append("")
        md.append("---")
        md.append("")
        
        # Recent updates
        md.append("## 🔄 Recent Updates")
        md.append("")
        md.append("| Timestamp | Entries Added |")
        md.append("|:----------|:--------------:|")
        
        # Group by timestamp
        by_timestamp = {}
        for entry in sorted_entries:
            if entry.timestamp not in by_timestamp:
                by_timestamp[entry.timestamp] = 0
            by_timestamp[entry.timestamp] += 1
        
        # Show last 5 timestamps
        recent_timestamps = sorted(by_timestamp.items(), reverse=True)[:5]
        for timestamp, count in recent_timestamps:
            md.append(f"| {timestamp} | {count} |")
        
        md.append("")
        md.append("---")
        md.append("")
        md.append("## 📝 Notes")
        md.append("")
        md.append("- ✅ Leaderboard is automatically updated via GitHub Actions (daily at 00:00 UTC)")
        md.append("- ✅ Rankings are based on time per step (lower is better)")
        md.append("- ✅ Only successful benchmarks are included")
        md.append("- ✅ Results are sorted by performance (fastest first)")
        md.append("- ✅ Historical data is preserved in `benchmark_history.json`")
        md.append("")
        md.append("### 🔧 Manual Update")
        md.append("")
        md.append("To manually update the leaderboard:")
        md.append("")
        md.append("```bash")
        md.append("# Run benchmarks")
        md.append("python -m zkaedi_prime_engine.benchmark")
        md.append("")
        md.append("# Generate leaderboard")
        md.append("python -m zkaedi_prime_engine.leaderboard")
        md.append("```")
        md.append("")
        md.append("Or trigger the workflow manually:")
        md.append("- Go to [Actions](https://github.com/zkaedii/h-benchmark/actions)")
        md.append("- Select 'Benchmark Leaderboard' workflow")
        md.append("- Click 'Run workflow'")
        md.append("")
        md.append("*This leaderboard is maintained automatically. For manual updates, see the benchmark suite.*")
        
        return "\n".join(md)
    
    def _empty_leaderboard(self) -> str:
        """Generate empty leaderboard template."""
        return """# 🏆 ZKAEDI PRIME Engine - Benchmark Leaderboard

**Real-time performance rankings** - Updated automatically via GitHub Actions

*No benchmark results yet. Run the benchmark suite to populate the leaderboard.*

## How to Update

1. Run the benchmark suite:
   ```bash
   python -m zkaedi_prime_engine.benchmark
   ```

2. Generate leaderboard:
   ```bash
   python -m zkaedi_prime_engine.leaderboard
   ```

3. The leaderboard will be automatically updated via GitHub Actions.

---

*Last updated: {timestamp}*
""".format(timestamp=time.strftime('%Y-%m-%d %H:%M:%S UTC'))
    
    def save(self):
        """Save leaderboard to file."""
        content = self.generate_leaderboard()
        self.output_file.write_text(content, encoding='utf-8')
    
    def run_and_update(self, quick: bool = False):
        """Run benchmarks and update leaderboard.
        
        Args:
            quick: If True, run only quick benchmarks (faster)
        """
        print("Running benchmarks for leaderboard...")
        benchmark = ZKAEDIPrimeBenchmark()
        
        if quick:
            # Quick benchmarks for CI/CD
            print("Running quick benchmarks...")
            results = benchmark.benchmark_scalability(qubit_counts=[2, 3, 4], timesteps=10)
            benchmark.results = results
        else:
            # Full benchmark suite
            benchmark.run_all(parallel=True)
        
        # Load existing history first
        self._load_history()
        
        # Add to leaderboard
        self.add_results(benchmark.results)
        
        # Save
        self.save()
        
        # Save history
        self.save_history()
        
        print(f"Leaderboard updated: {self.output_file}")


def main():
    """Main function to generate leaderboard."""
    import sys
    
    quick = '--quick' in sys.argv or '--ci' in sys.argv
    
    leaderboard = BenchmarkLeaderboard()
    leaderboard.run_and_update(quick=quick)
    leaderboard.save_history()


if __name__ == "__main__":
    main()
