"""Task 3b - prove a 20-token shared prefix costs 0 new blocks the second time."""
import _bootstrap  # noqa: F401

from paged_attention.mock_llm import MockLLM
from paged_attention.prefix_cache import PrefixCachingAllocator

BLOCK_SIZE = 4


def main() -> None:
    llm = MockLLM()
    alloc = PrefixCachingAllocator(num_blocks=256, block_size=BLOCK_SIZE)

    shared_prefix = [llm.token(request_id=1, position=i) for i in range(20)]

    # Two requests, same 20-token prefix, different continuations.
    req_a = shared_prefix + [llm.token(request_id=1, position=i) for i in range(20, 28)]
    req_b = shared_prefix + [llm.token(request_id=2, position=i) for i in range(20, 28)]

    result_a = alloc.admit(req_a)
    result_b = alloc.admit(req_b)

    prefix_blocks = 20 // BLOCK_SIZE
    # Of req_b's blocks, the first `prefix_blocks` cover the shared 20 tokens.
    new_for_prefix = sum(
        1 for i in range(prefix_blocks)
        if result_b["page_table"][i] not in result_a["page_table"][:prefix_blocks]
    )

    print(f"block_size={BLOCK_SIZE}, shared prefix = 20 tokens ({prefix_blocks} blocks)\n")
    print(f"request A: {result_a['new_allocations']} blocks allocated (cold cache)")
    print(f"request B: {result_b['reused_blocks']} blocks reused, "
          f"{result_b['new_allocations']} allocated (only its own 8-token tail)")
    print(f"new physical blocks for the shared 20-token prefix on B: {new_for_prefix}")

    assert new_for_prefix == 0, "prefix was re-allocated instead of shared"
    # The shared prefix blocks are literally the same physical blocks.
    assert result_b["page_table"][:prefix_blocks] == result_a["page_table"][:prefix_blocks]
    print("\nOK: the 20-token prefix was reused with zero new allocations.")


if __name__ == "__main__":
    main()
