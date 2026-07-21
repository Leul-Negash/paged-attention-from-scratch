"""Task 2c - craft a guaranteed deadlock, detect and recover in one step."""
import _bootstrap  # noqa: F401

import random

from paged_attention.deadlock import DeadlockManager


def crafted_two_cycle() -> None:
    print("crafted deadlock: A holds 7 wants 12, B holds 12 wants 7")
    dm = DeadlockManager()
    dm.add_request(rid=ord("A"), holds={7}, waiting_for=12, tokens=5)
    dm.add_request(rid=ord("B"), holds={12}, waiting_for=7, tokens=3)

    cycle = dm.detect()
    print(f"  detected cycle: {[chr(n) for n in cycle]}")

    actions = dm.recover_step()  # same time step
    for a in actions:
        print(f"  evicted {chr(a['victim'])} (held most blocks / fewest tokens), "
              f"freed {a['freed']}")

    assert dm.detect() is None, "still deadlocked after recovery"
    print(f"  post-recovery cycle: {dm.detect()}   "
          f"A now holds {sorted(dm.holds[ord('A')])}\n")


def crafted_three_cycle() -> None:
    print("crafted 3-way deadlock: A->B->C->A")
    dm = DeadlockManager()
    dm.add_request(1, holds={1, 2, 3}, waiting_for=10, tokens=8)  # most blocks
    dm.add_request(2, holds={10}, waiting_for=20, tokens=4)
    dm.add_request(3, holds={20}, waiting_for=1, tokens=6)

    print(f"  detected cycle: {dm.detect()}")
    actions = dm.recover_step()
    print(f"  evicted request {actions[0]['victim']} (holds 3 blocks, the most)")
    assert actions[0]["victim"] == 1
    assert dm.detect() is None
    print("  recovered in one step\n")


def random_stress() -> None:
    """Random hold/wait graphs must always fully recover, never hang."""
    rng = random.Random(0)
    for trial in range(2000):
        dm = DeadlockManager()
        n = rng.randint(2, 8)
        blocks = list(range(n))
        rng.shuffle(blocks)
        for i in range(n):
            wait = blocks[(i + 1) % n] if rng.random() < 0.8 else None
            dm.add_request(i, holds={blocks[i]}, waiting_for=wait,
                           tokens=rng.randint(0, 20))
        dm.recover_step()
        assert dm.detect() is None, f"trial {trial} left a cycle"
    print("random stress: 2000 graphs, every one fully recovered, zero hangs\n")


def main() -> None:
    crafted_two_cycle()
    crafted_three_cycle()
    random_stress()
    print("OK: deadlocks detected and broken in the same step they form.")


if __name__ == "__main__":
    main()
