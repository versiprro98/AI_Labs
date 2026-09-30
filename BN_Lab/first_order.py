"""First-order autoregressive language model (bigram / Markov chain).

Models P(X_t | X_{t-1}) using transition counts stored in plain dicts.
No ML libraries or pretrained models are used.
"""
import random
from collections import defaultdict

START, END = "<START>", "<END>"

DATASET = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the dog sat on the mat",
    "the dog ran to the park",
    "the cat ran to the park",
    "the dog sat on the rug",
]


def tokenise(sentence):
    """Lower-case, split on whitespace, add <START> and <END>."""
    return [START] + sentence.lower().split() + [END]


class FirstOrderLM:
    def __init__(self):
        # counts[prev][next] = C(prev, next)
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probs = {}

    def fit(self, sentences):
        """sentences: list of token lists (already including START/END)."""
        for tokens in sentences:
            for prev, nxt in zip(tokens[:-1], tokens[1:]):
                self.counts[prev][nxt] += 1
        # P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)
        self.probs = {}
        for prev, nxt_counts in self.counts.items():
            total = sum(nxt_counts.values())
            self.probs[prev] = {w: c / total for w, c in nxt_counts.items()}
        return self

    def distribution(self, prev):
        """Return {next_token: prob}, or {} if `prev` was never seen."""
        return dict(self.probs.get(prev, {}))

    def show(self, prev):
        dist = self.distribution(prev)
        if not dist:
            print(f"P(next | {prev}) : no observed transitions")
            return
        for w, p in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"P({w} | {prev}) = {p:.3f}")

    def most_probable(self, prev):
        dist = self.distribution(prev)
        if not dist:
            return None
        # sorted by (-prob, token) so ties are broken deterministically
        return sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    def sample_next(self, prev, rng=random):
        dist = self.distribution(prev)
        if not dist:
            return None
        words = list(dist.keys())
        weights = list(dist.values())
        return rng.choices(words, weights=weights, k=1)[0]

    def generate(self, mode="sample", max_len=30, rng=random):
        """Generate one sentence. mode: 'sample' or 'greedy'."""
        out, prev = [], START
        for _ in range(max_len):
            nxt = self.sample_next(prev, rng) if mode == "sample" else self.most_probable(prev)
            if nxt is None or nxt == END:  # unseen context or END => stop
                break
            out.append(nxt)
            prev = nxt
        return " ".join(out)

    def n_parameters(self):
        """Number of non-zero CPT entries (distinct observed bigrams)."""
        return sum(len(d) for d in self.probs.values())


def check_normalisation(probs, tol=1e-9):
    """Check that sum_v P(v | context) == 1 for every context."""
    ok = True
    for ctx, dist in probs.items():
        total = sum(dist.values())
        flag = "OK" if abs(total - 1.0) < tol else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {str(ctx):<28} sum = {total:.6f}  {flag}")
    return ok


if __name__ == "__main__":
    data = [tokenise(s) for s in DATASET]
    lm = FirstOrderLM().fit(data)
    for w in [START, "the", "cat", "dog", "sat", "ran"]:
        lm.show(w)
        print()
    print("Normalisation check:")
    check_normalisation(lm.probs)
    random.seed(0)
    print("\nGenerated:")
    for _ in range(5):
        print(" ", lm.generate())
