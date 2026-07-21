"""A stand-in for a real model.

We never run an LLM here. All we need from generation is a stream of tokens
that is *deterministic* in the request id and the token position, so that a
request restarted from scratch (recompute) or reloaded from a swap produces
exactly the same output it would have if it had never been interrupted.
That property is what makes the byte-identical checks in Task 2a and the
prefix-content hashing in Task 3b meaningful.
"""


class MockLLM:
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size

    def token(self, request_id: int, position: int) -> int:
        # A small integer hash (splitmix-style) mixed from (id, position).
        # Python's built-in hash() is salted per process, so we roll our own
        # to stay stable across runs.
        h = (request_id * 0x9E3779B1 + position * 0x85EBCA6B + 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        h = (h * 0x7FEB352D) & 0xFFFFFFFF
        h ^= h >> 15
        h = (h * 0x846CA68B) & 0xFFFFFFFF
        h ^= h >> 16
        return h % self.vocab_size

    def sequence(self, request_id: int, length: int) -> list[int]:
        return [self.token(request_id, i) for i in range(length)]
