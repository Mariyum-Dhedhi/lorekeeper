"""Retrieval metrics computed at the document (source file) level."""


def unique_in_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def hit_at_k(retrieved: list[str], gold: list[str]) -> float:
    return 1.0 if set(retrieved) & set(gold) else 0.0


def recall_at_k(retrieved: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(set(gold))


def mrr(retrieved: list[str], gold: list[str]) -> float:
    for rank, doc in enumerate(unique_in_order(retrieved), start=1):
        if doc in gold:
            return 1.0 / rank
    return 0.0


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None
