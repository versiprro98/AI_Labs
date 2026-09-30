"""Second-order autoregressive language model: P(X_t | X_{t-2}, X_{t-1}).

Uses counts of observed triples. Contexts are pairs of tokens. The first
word is conditioned on (<START>, <START>) so the sentence start is a context.
"""
import random
from collections import defaultdict

from first_order import START, END, DATASET, tokenise, check_normalisation


class SecondOrderLM:
    def __init__(self):
        # counts[(w1, w2)][w3] = C(w1, w2, w3)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probs = {}

    def fit(self, sentences):
        for tokens in sentences:
            padded = [START] + tokens  # (<START>, <START>) is the initial context
            for a, b, c in zip(padded[:-2], padded[1:-1], padded[2:]):
                self.counts[(a, b)][c] += 1
        self.probs = {}
        for ctx, nxt_counts in self.counts.items():
            total = sum(nxt_counts.values())
            self.probs[ctx] = {w: c / total for w, c in nxt_counts.items()}
        return self

    def distribution(self, ctx):
        return dict(self.probs.get(tuple(ctx), {}))

    def show(self, ctx):
        dist = self.distribution(ctx)
        if not dist:
            print(f"P(next | {ctx}) : unseen context")
            return
        for w, p in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"P({w} | {ctx[0]}, {ctx[1]}) = {p:.3f}")

    def most_probable(self, ctx):
        dist = self.distribution(ctx)
        if not dist:
            return None
        return sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def sample_next(self, ctx, rng=random):
        dist = self.distribution(ctx)
        if not dist:
            return None
        return rng.choices(list(dist), weights=list(dist.values()), k=1)[0]

    def generate(self, mode="sample", max_len=30, rng=random):
        out, ctx = [], (START, START)
        for _ in range(max_len):
            nxt = self.sample_next(ctx, rng) if mode == "sample" else self.most_probable(ctx)
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            ctx = (ctx[1], nxt)
        return " ".join(out)

    def n_parameters(self):
        return sum(len(d) for d in self.probs.values())


if __name__ == "__main__":
    data = [tokenise(s) for s in DATASET]
    lm = SecondOrderLM().fit(data)
    for ctx in [(START, START), (START, "the"), ("the", "cat"), ("cat", "sat"), ("sat", "on"), ("on", "the")]:
        lm.show(ctx)
        print()
    print("Normalisation check:")
    check_normalisation(lm.probs)
    random.seed(0)
    for _ in range(5):
        print(" ", lm.generate())
