"""Task 2c - craft a guaranteed deadlock, detect and recover in one step.

Also runs a live simulation where the cycle-detection pass fires on *every*
time step, so deadlocks that emerge naturally are broken as they form.
"""
import _bootstrap  # noqa: F401

import random

from paged_attention.deadlock import DeadlockManager

SIM_STEPS = 300
INJECT_AT_STEP = 150


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


def _inject_guaranteed_cycle(dm: DeadlockManager) -> tuple[int, int]:
    """Make two live requests wait on each other's block."""
    a, b = sorted(dm.holds)[:2]
    dm.waiting_for[a] = next(iter(dm.holds[b]))
    dm.waiting_for[b] = next(iter(dm.holds[a]))
    return a, b


def stepped_simulation() -> None:
    """Detection runs on every step of a live, contended simulation."""
    rng = random.Random(11)
    dm = DeadlockManager()
    num_requests, spare = 12, 3
    for i in range(num_requests):
        dm.add_request(i, holds={i}, tokens=rng.randint(0, 40))
    dm.free = set(range(num_requests, num_requests + spare))

    cycles_broken = 0
    injected_caught = False

    for step in range(SIM_STEPS):
        # Requests that are not blocked make progress.
        for rid, waiting in dm.waiting_for.items():
            if waiting is None:
                dm.tokens[rid] += 1

        # Some idle requests start waiting on a block another request holds.
        ids = list(dm.holds)
        for rid in ids:
            if dm.waiting_for[rid] is None and rng.random() < 0.3:
                held_elsewhere = [b for other in ids if other != rid
                                  for b in dm.holds[other]]
                if held_elsewhere:
                    dm.waiting_for[rid] = rng.choice(held_elsewhere)

        # A waiter whose block has come free simply takes it.
        for rid, block in list(dm.waiting_for.items()):
            if block is not None and block in dm.free:
                dm.free.discard(block)
                dm.holds[rid].add(block)
                dm.waiting_for[rid] = None

        if step == INJECT_AT_STEP:
            a, b = _inject_guaranteed_cycle(dm)

        actions = dm.recover_step()   # <-- every single time step
        cycles_broken += len(actions)

        if step == INJECT_AT_STEP:
            assert actions, "injected cycle was not detected in its own step"
            injected_caught = True
            print(f"stepped simulation: guaranteed cycle injected between "
                  f"requests {a} and {b} at step {step}")
            print(f"  detected and broken in that same step "
                  f"(evicted {actions[0]['victim']})")

        # No cycle may survive the step in which it was detected.
        assert dm.detect() is None, f"a cycle survived step {step}"

        # Evicted requests are rescheduled with a fresh block.
        for act in actions:
            if dm.free:
                dm.add_request(act["victim"], holds={dm.free.pop()}, tokens=0)

    assert injected_caught
    print(f"  detection ran on all {SIM_STEPS} steps, "
          f"{cycles_broken} cycles broken, none ever survived a step\n")


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
    stepped_simulation()
    random_stress()
    print("OK: deadlocks detected and broken in the same step they form.")


if __name__ == "__main__":
    main()
