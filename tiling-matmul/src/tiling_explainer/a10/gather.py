from collections.abc import Callable, Mapping
from typing import Protocol

MAX_REPLACEMENTS = 6

Call = dict[str, object]


class Pending(Protocol):
    def get(self) -> Call: ...


class TooManyRefusals(RuntimeError):
    pass


def gather(
    starters: Mapping[str, Callable[[], Pending]],
    wanted: Mapping[str, int],
    max_replacements: int = MAX_REPLACEMENTS,
) -> tuple[dict[str, list[Call]], list[Call]]:
    pending = [(kind, starters[kind]()) for kind, count in wanted.items() for _ in range(count)]
    accepted: dict[str, list[Call]] = {kind: [] for kind in wanted}
    refusals: list[Call] = []
    while pending:
        kind, call = pending.pop(0)
        result = call.get()
        if "refused" not in result:
            accepted[kind].append(result)
            continue
        refusals.append({**result, "kind": kind})
        if len(refusals) > max_replacements:
            reasons = [str(refusal["refused"]) for refusal in refusals]
            raise TooManyRefusals(
                f"{len(refusals)} calls refused, limit {max_replacements}: {reasons}"
            )
        pending.append((kind, starters[kind]()))
    return accepted, refusals
