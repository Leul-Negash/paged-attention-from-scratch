"""Run every task end to end. Task 1a prints a per-step memory map and is long."""
import subprocess
import sys
from pathlib import Path

TASKS = [
    ("1a", "Naive contiguous allocator", "run/task1a_naive.py"),
    ("1b", "PagedAllocator leak/correctness test", "run/task1b_paged.py"),
    ("1c", "Naive vs Paged head-to-head", "run/task1c_compare.py"),
    ("1d", "Correctness harness + injected bugs", "run/task1d_harness.py"),
    ("2a", "Preemption engine (recompute + swap)", "run/task2a_preemption.py"),
    ("2b", "SLA scheduler", "run/task2b_scheduler.py"),
    ("2c", "Deadlock detection and recovery", "run/task2c_deadlock.py"),
    ("3a", "Copy-on-write", "run/task3a_cow.py"),
    ("3b", "Prefix caching", "run/task3b_prefix.py"),
    ("3c", "Distributed page table", "run/task3c_distributed.py"),
]

ROOT = Path(__file__).parent


def main() -> None:
    only = sys.argv[1:]  # e.g. `python run_all.py 2a 2b`
    for tag, title, path in TASKS:
        if only and tag not in only:
            continue
        print("\n" + "=" * 72)
        print(f"TASK {tag}  -  {title}")
        print("=" * 72)
        subprocess.run([sys.executable, str(ROOT / path)], check=True)


if __name__ == "__main__":
    main()
