"""Runs every experiment in the lab and writes results to results/."""
import os
import random

from first_order import START, END, DATASET, tokenise, FirstOrderLM, check_normalisation
from second_order import SecondOrderLM

os.makedirs("results", exist_ok=True)
data = [tokenise(s) for s in DATASET]
fo = FirstOrderLM().fit(data)
so = SecondOrderLM().fit(data)
vocab = sorted({w for s in data for w in s})
log = []


def out(line=""):
    print(line)
    log.append(line)


# ---- Part IV / Q3: CPT for selected words -------------------------------
out("=== First-order CPTs (Q3) ===")
for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]:
    dist = fo.distribution(w)
    out(f"P(next | {w}): " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))))

out("\nZero-probability transitions (for the five required words), vocabulary incl. <END>:")
targets = [w for w in vocab if w != START]
for w in ["the", "cat", "dog", "sat", "ran"]:
    zeros = [v for v in targets if fo.probs[w].get(v, 0) == 0]
    out(f"  {w}: {zeros}")

# ---- Part VII: normalisation tests -------------------------------------
out("\n=== Normalisation test, first order ===")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ok1 = check_normalisation(fo.probs)
out(buf.getvalue().rstrip())
out(f"All first-order contexts sum to 1: {ok1}")
out("\n=== Normalisation test, second order ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ok2 = check_normalisation(so.probs)
out(buf.getvalue().rstrip())
out(f"All second-order contexts sum to 1: {ok2}")

# ---- Part VIII: next-word prediction -----------------------------------
out("\n=== Most probable next word (first order) ===")
for w in [START, "the", "cat", "dog", "sat", "on", "ran", "to"]:
    out(f"argmax P(. | {w}) = {fo.most_probable(w)}")

# ---- Part IX: generate 20 sentences ------------------------------------
random.seed(42)
sampled = [fo.generate("sample") for _ in range(20)]
out("\n=== 20 sampled sentences (first order, seed 42) ===")
for s in sampled:
    out("  " + s)
with open("results/generated_first_order.txt", "w") as f:
    f.write("\n".join(sampled) + "\n")

# ---- Part X: greedy vs sampling ----------------------------------------
random.seed(1)
out("\n=== Greedy vs sampling, first order (5 each) ===")
greedy5 = [fo.generate("greedy") for _ in range(5)]
sample5 = [fo.generate("sample") for _ in range(5)]
out("Greedy:   " + " | ".join(greedy5))
out("Sampling: " + " | ".join(sample5))
out(f"Distinct greedy: {len(set(greedy5))}, distinct sampled: {len(set(sample5))}")

# ---- Part XI/XII: second order -----------------------------------------
random.seed(42)
sampled2 = [so.generate("sample") for _ in range(20)]
out("\n=== 20 sampled sentences (second order, seed 42) ===")
for s in sampled2:
    out("  " + s)
with open("results/generated_second_order.txt", "w") as f:
    f.write("\n".join(sampled2) + "\n")
out("\nSecond-order greedy: " + so.generate("greedy"))

out("\n=== Second-order CPTs (selected contexts) ===")
for ctx in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"), ("cat", "sat"),
            ("cat", "ran"), ("dog", "sat"), ("dog", "ran"), ("sat", "on"), ("on", "the"), ("to", "the")]:
    dist = so.distribution(ctx)
    out(f"P(next | {ctx[0]}, {ctx[1]}): " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))))

# ---- Part XIII: comparison ---------------------------------------------
training = set(DATASET)
def novel(sents):
    return sorted({s for s in sents if s not in training})

# larger samples for diversity statistics
random.seed(7)
big1 = [fo.generate("sample") for _ in range(1000)]
big2 = [so.generate("sample") for _ in range(1000)]
n_ctx1 = len(fo.probs)
n_ctx2 = len(so.probs)
possible_ctx1 = len(vocab) - 1          # every token except <END> could precede
possible_ctx2 = (len(vocab) - 1) ** 2   # ordered pairs of non-END tokens
out("\n=== Comparison ===")
out(f"Vocabulary size (incl. START/END): {len(vocab)}")
out(f"Observed contexts:            first={n_ctx1}, second={n_ctx2}")
out(f"Non-zero parameters:          first={fo.n_parameters()}, second={so.n_parameters()}")
out(f"Possible contexts (upper bound): first={possible_ctx1}, second={possible_ctx2}")
out(f"Contexts never observed:      first={possible_ctx1 - n_ctx1}, second={possible_ctx2 - n_ctx2}")
out(f"Full CPT size (contexts x next tokens): first={possible_ctx1 * (len(vocab)-1)}, second={possible_ctx2 * (len(vocab)-1)}")
out(f"Distinct sentences in 1000 samples: first={len(set(big1))}, second={len(set(big2))}")
out(f"Novel (not in training) among distinct: first={len(novel(set(big1)))}, second={len(novel(set(big2)))}")
out("Novel first-order examples: " + "; ".join(novel(set(big1))[:6]))
out("Novel second-order examples: " + ("; ".join(novel(set(big2))[:6]) or "none"))
out("Training sentences reproduced verbatim: "
    f"first={sum(s in training for s in big1)/10:.1f}%, second={sum(s in training for s in big2)/10:.1f}%")

# ---- Q7 edge case: unseen word -----------------------------------------
out("\n=== Q7: unseen context ===")
out(f"fo.distribution('bird') = {fo.distribution('bird')}; most_probable = {fo.most_probable('bird')}")

with open("results/results.txt", "w") as f:
    f.write("\n".join(log) + "\n")
