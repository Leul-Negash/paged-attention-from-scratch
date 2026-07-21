"""Task 2a - preemption engine on 500 oversubscribed requests."""
import _bootstrap  # noqa: F401

from paged_attention.preemption import PreemptionEngine
from paged_attention.workload import generate_workload

NUM_BLOCKS = 128
BLOCK_SIZE = 16       # capacity = 2048 tokens
THRESHOLD_T = 24      # below -> recompute, at or above -> swap
N = 500


def swap_resume_demo() -> None:
    """A focused proof that a swapped-and-resumed request is byte-identical.

    Generate half a request, force it through swap mode, then let it finish and
    compare against the sequence it would have produced untouched.
    """
    engine = PreemptionEngine(num_blocks=8, block_size=4, threshold_T=1)
    req = generate_workload(1, seed=99)[0]
    req.actual_length = 30
    reference = [engine.alloc.llm.token(req.id, i) for i in range(30)]

    engine.alloc.admit(req)
    engine.running[req.id] = req
    for _ in range(15):
        engine._grow(req)

    engine._evict(req)                 # tokens_generated=15 >= T -> swap mode
    assert req.id in engine.swapped, "expected swap, not recompute"

    engine._admit(req)                 # reload from the swap store
    while not req.done:
        engine._grow(req)

    print("swap-resume demo: interrupted at token 15, resumed to 30")
    print(f"  byte-identical to never-interrupted: {req.tokens == reference}\n")
    assert req.tokens == reference


def main() -> None:
    swap_resume_demo()

    requests = generate_workload(N, seed=42, mean_service=80.0)
    # Compress arrivals so the system is genuinely oversubscribed, not just busy.
    for r in requests:
        r.arrival = r.id // 40

    engine = PreemptionEngine(NUM_BLOCKS, BLOCK_SIZE, THRESHOLD_T)
    stats = engine.run(requests)

    print(f"Preemption engine  |  {NUM_BLOCKS} blocks x {BLOCK_SIZE} "
          f"= {stats['capacity_tokens']} tokens  |  T={THRESHOLD_T}\n")
    print(f"requests completed       : {stats['completed']} / {N}")
    print(f"peak concurrent demand   : {stats['peak_demand']} tokens")
    print(f"oversubscription ratio   : {stats['oversubscription']:.1f}x  "
          f"(requirement: >= 3x)")
    print(f"recompute evictions      : {stats['recompute_evictions']}")
    print(f"swap evictions           : {stats['swap_evictions']}")
    print(f"byte-identical mismatches: {stats['mismatches']}  (target: 0)")

    assert stats["oversubscription"] >= 3.0, "workload was not 3x oversubscribed"
    assert stats["mismatches"] == 0, "a resumed request diverged"
    assert stats["completed"] == N
    print("\nOK: 3x+ oversubscribed, every request finished, zero divergence.")


if __name__ == "__main__":
    main()
