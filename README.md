# PagedAttention from Scratch

A pure-Python simulation of how an LLM inference server manages KV-cache memory:
physical blocks, per-request page tables, preemption, scheduling, deadlock
recovery, copy-on-write, prefix caching and a multi-node page table. No GPU, no
ML libraries — everything is a Python data structure.

## The "mock LLM"

We never run a real model. Generation is a `MockLLM` (`paged_attention/mock_llm.py`)
that returns a token deterministic in `(request_id, position)`. That determinism
is what makes the hard checks verifiable: a request that is recomputed, swapped
out and reloaded, or rescheduled onto another node produces **byte-identical**
output to one that was never touched, and two requests sharing a prompt hash to
the same block content.

## Layout

```
paged_attention/            core library, one module per concept
  mock_llm.py               deterministic token generator
  request.py                Request model (priority, deadline, token_budget)
  physical_blocks.py        the global free pool  (allocate / free)
  page_table.py             logical block index -> physical block id
  paged_allocator.py        on-demand allocation, ties the two together
  naive_allocator.py        the contiguous allocator PagedAttention replaces
  simulation.py             discrete-step drivers for naive and paged
  harness.py                Task 1d invariant checker
  preemption.py             Task 2a recompute / swap engine
  scheduler.py              Task 2b SLA scheduler
  deadlock.py               Task 2c wait-for graph + DFS cycle detection
  cow.py                    Task 3a copy-on-write with ref counts
  prefix_cache.py           Task 3b content-addressed prefix sharing
  distributed.py            Task 3c 4-node page table
run/                        one runnable script per task deliverable
run_all.py                  runs everything (optionally: `python run_all.py 2a 2b`)
```

## Running

```
python run_all.py            # all tasks
python run/task1c_compare.py # a single task
```

Each script is self-contained and prints its own results and assertions.

## What each task shows

| Task | Deliverable | Result |
|------|-------------|--------|
| 1a | Naive contiguous allocator on a 200-req Poisson workload | per-step memory map, ~46% internal fragmentation, rejections |
| 1b | PagedAllocator correctness | 200,000 requests (10k x 5 seeds x block sizes 4/8/16/32), zero leaks, no off-by-one |
| 1c | Naive vs Paged, same seed | Paged: 0 rejections vs 94, ~4% internal vs ~46%, deterministic |
| 1d | Correctness harness | disjointness held every step; double-free, use-after-free and boundary-miss injected **mid-run**, all caught, run survives |
| 2a | Preemption under 3x+ oversubscription | recompute + swap, all 500 finish, **0** byte-identical mismatches |
| 2b | SLA scheduler | priority + deadline order, budgets enforced exactly, **0** deadline misses, **0** orphaned blocks |
| 2c | Deadlock detection | DFS cycle detection O(V+E) run **every time step**; crafted 2- and 3-cycles broken in one step; 300-step live sim, no cycle survives its step; 2000 random graphs recover |
| 3a | Copy-on-write | N in {1,2,4,8}, reads share, writes clone (N-1 clones), no crossed streams |
| 3b | Prefix caching | 20-token shared prefix reused with **0** new blocks on the second request |
| 3c | Distributed page table | spill to freest node, hot-block migration, node 2 killed at step 500, **0** hangs |

## Notes

- Block size is a constructor parameter and works for any power of two.
- The naive and paged allocators run on the identical seeded workload so the
  head-to-head numbers are reproducible to the digit.
