"""B2: majority vote over parseable final answers; ties -> smallest original index; none parseable -> 0."""
from collections import Counter
from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        parsed = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = Counter(p for p in parsed if p is not None)
        if not counts:
            choices.append(0)
            continue
        top = max(counts.values())
        winners = {a for a, c in counts.items() if c == top}
        pick = 0
        for j, p in enumerate(parsed):
            if p in winners:
                pick = j
                break
        choices.append(pick)
    return choices
