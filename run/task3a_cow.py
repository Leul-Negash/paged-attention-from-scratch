"""Task 3a - copy-on-write across N in {1, 2, 4, 8} parallel completions."""
import _bootstrap  # noqa: F401

from paged_attention.cow import CoWAllocator

BLOCK_SIZE = 4
PROMPT_LEN = 18   # not a multiple of block_size -> last prompt block is shared+partial
GEN = 12


def run_for_n(n: int) -> None:
    alloc = CoWAllocator(num_blocks=256, block_size=BLOCK_SIZE)
    prompt = [alloc.llm.token(request_id=1000, position=i) for i in range(PROMPT_LEN)]
    seqs = alloc.fork(prompt, n)

    # Before any write, every prompt block is shared by all N completions.
    prompt_blocks = set(seqs[0].page_table)
    assert all(alloc.blocks[b].ref_count == n for b in prompt_blocks)

    def invariant_holds(writer, written_block) -> bool:
        # No other completion may reference the block we just wrote to.
        return all(written_block not in s.page_table
                   for s in seqs if s is not writer)

    for k, seq in enumerate(seqs):
        completion_key = 2000 + n * 10 + k     # unique token stream per completion
        for pos in range(GEN):
            token = alloc.llm.token(completion_key, pos)
            written = alloc.append(seq, token)
            assert invariant_holds(seq, written), "completions crossed streams"

    # Each completion reads back its own generated tokens, untouched by others.
    for k, seq in enumerate(seqs):
        completion_key = 2000 + n * 10 + k
        for pos in range(GEN):
            logical, offset = divmod(PROMPT_LEN + pos, BLOCK_SIZE)
            block = alloc.blocks[seq.page_table[logical]]
            assert block.data[offset] == alloc.llm.token(completion_key, pos)

    used_before = alloc.num_used
    for seq in seqs:
        alloc.cancel(seq)
    assert alloc.num_used == 0, "blocks leaked after cancellation"

    print(f"N={n}:  shared prompt blocks={len(prompt_blocks)}  "
          f"clones-on-write={alloc.clones}  peak blocks={used_before}  "
          f"freed cleanly=yes")


def main() -> None:
    print(f"Copy-on-write  |  block_size={BLOCK_SIZE}  prompt={PROMPT_LEN} tokens\n")
    for n in (1, 2, 4, 8):
        run_for_n(n)
    print("\nOK: reads share, writes clone, no completion saw another's tokens.")


if __name__ == "__main__":
    main()
