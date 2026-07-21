"""Task 3c - spill, migration, and surviving a node going offline at step 500."""
import _bootstrap  # noqa: F401

import random

from paged_attention.distributed import DistributedAllocator, DistributedEngine, DistributedRequest


def spill_demo() -> None:
    alloc = DistributedAllocator(num_nodes=4, blocks_per_node=2)
    home = 0
    locs = [alloc.allocate(home) for _ in range(6)]  # home holds only 2
    landed = {loc[0] for loc in locs}
    print(f"spill demo: 6 allocations from home node 0 -> landed on nodes {sorted(landed)}")
    assert landed != {0}, "allocations never spilled off the home node"


def migration_demo() -> None:
    alloc = DistributedAllocator(num_nodes=4, blocks_per_node=4)
    # Force a block onto a remote node, then migrate it to home 0.
    remote = alloc.allocate(home=1)
    alloc.nodes[remote[0]].data[remote[1]] = [111, 222]
    moved = alloc.migrate(remote, home=0)
    print(f"migration demo: block at {remote} accessed >3x -> moved to {moved}")
    assert moved[0] == 0, "hot remote block did not migrate home"
    assert alloc.nodes[0].data[moved[1]] == [111, 222], "data lost in migration"


def failure_sim() -> None:
    rng = random.Random(3)
    requests = []
    for i in range(80):
        requests.append(DistributedRequest(
            req_id=i,
            arrival=i * 15,
            actual_length=rng.randint(200, 350),
            home=(i // 8) % 4,   # bursts of 8 share a home -> hot nodes spill
        ))

    engine = DistributedEngine(num_nodes=4, blocks_per_node=128, block_size=16)
    stats = engine.run(requests, fail_step=500, fail_node=2)

    print("\nfailure sim: 4 nodes, node 2 killed at step 500")
    print(f"  requests completed        : {stats['completed']} / {len(requests)}")
    print(f"  spills (home was full)    : {stats['spills']}")
    print(f"  hot-block migrations      : {stats['migrations']}")
    print(f"  rescheduled after failure : {stats['preempted_by_failure']}")
    print(f"  output mismatches         : {stats['mismatches']}  (target: 0)")

    assert stats["completed"] == len(requests), "a request hung after node failure"
    assert stats["mismatches"] == 0, "output diverged after reschedule"
    assert stats["preempted_by_failure"] > 0, "no request was on node 2 at failure"
    assert stats["spills"] > 0 and stats["migrations"] > 0, "spill/migration not exercised"


def main() -> None:
    spill_demo()
    migration_demo()
    failure_sim()
    print("\nOK: allocations spill, hot blocks migrate, node loss recovers with no hang.")


if __name__ == "__main__":
    main()
