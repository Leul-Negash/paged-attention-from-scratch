"""Task 2b - SLA scheduler on 500 requests with tight deadlines and budgets."""
import _bootstrap  # noqa: F401

from paged_attention.scheduler import SLAScheduler
from paged_attention.workload import add_sla_fields, generate_workload

NUM_BLOCKS = 512
BLOCK_SIZE = 16
N = 500


def main() -> None:
    requests = generate_workload(N, seed=42, mean_service=50.0)
    add_sla_fields(requests, seed=7)

    sched = SLAScheduler(NUM_BLOCKS, BLOCK_SIZE)
    stats = sched.run(requests)

    # Every budget-capped request must have stopped at exactly its budget.
    for r in requests:
        if r.token_budget < r.actual_length:
            assert r.tokens_generated == r.token_budget, "budget not enforced exactly"

    print(f"SLA scheduler  |  {NUM_BLOCKS} blocks x {BLOCK_SIZE}  |  {N} requests\n")
    print(f"requests completed  : {stats['completed']} / {N}")
    print(f"deadline misses     : {stats['deadline_misses']}  (target: 0)")
    print(f"budget terminations : {stats['budget_terminations']}")
    print(f"orphaned blocks     : {stats['orphaned_blocks']}  (must be 0)")

    assert stats["orphaned_blocks"] == 0, "blocks leaked on termination"
    assert stats["completed"] == N
    assert stats["deadline_misses"] == 0, "missed a deadline with capacity free"
    print("\nOK: zero deadline misses, budgets enforced exactly, zero orphans.")


if __name__ == "__main__":
    main()
