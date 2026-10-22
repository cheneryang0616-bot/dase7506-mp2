"""B3: first candidate whose final answer parses; else 0."""
from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []
    for row in bank:
        pick = 0
        for j, c in enumerate(row["candidates"]):
            if parse_answer(c["text"]) is not None:
                pick = j
                break
        choices.append(pick)
    return choices
