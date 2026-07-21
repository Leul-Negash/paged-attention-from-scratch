import random

from .request import Priority, Request


def generate_workload(
    n: int,
    seed: int,
    arrival_rate: float = 0.7,
    mean_service: float = 60.0,
    max_len_range: tuple[int, int] = (10, 512),
) -> list[Request]:
    """A reproducible stream of requests.

    Arrivals follow a Poisson process (exponential inter-arrival gaps) and
    service length is drawn from an exponential distribution, clipped to the
    request's own max_length. Same seed => identical stream, which is what
    lets the naive and paged allocators be compared head-to-head.
    """
    rng = random.Random(seed)
    requests = []
    clock = 0.0
    for i in range(n):
        clock += rng.expovariate(arrival_rate)
        max_length = rng.randint(*max_len_range)
        service = rng.expovariate(1.0 / mean_service)
        actual_length = max(1, min(max_length, int(service) + 1))
        requests.append(
            Request(
                id=i,
                arrival=int(clock),
                max_length=max_length,
                actual_length=actual_length,
            )
        )
    return requests


def add_sla_fields(requests: list[Request], seed: int) -> list[Request]:
    """Attach random priorities, deadlines and token budgets (Task 2b)."""
    rng = random.Random(seed)
    for r in requests:
        r.priority = Priority(rng.choice([1, 2, 3]))
        r.deadline = rng.randint(1, 12)              # tight queue-wait budget
        r.token_budget = rng.randint(5, r.max_length)  # may cut generation short
    return requests
