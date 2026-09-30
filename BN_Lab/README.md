# AI Laboratory: Bayesian Networks and Autoregressive Language Models

## Files

| File | Purpose |
|---|---|
| `first_order.py` | First-order model, P(X_t \| X_{t-1}), counts, CPT, greedy/sampling generation, normalisation check |
| `second_order.py` | Second-order model, P(X_t \| X_{t-2}, X_{t-1}), counts of triples |
| `run_lab.py` | Runs every experiment and writes `results/` |
| `results/results.txt` | CPTs, normalisation tests, predictions, generated text, comparison |
| `results/generated_first_order.txt`, `results/generated_second_order.txt` | 20 sampled sentences from each model |

Run with `python run_lab.py` (Python 3, standard library only, no ML libraries).

---

## Answers

### Question 1: Why is the chain-rule decomposition useful for generating text?
The chain rule turns one huge joint distribution over whole sentences into a sequence of small conditional distributions, each over a single next token. A sentence can then be generated one token at a time: sample X1 from P(X1), then X2 from P(X2 | X1), and so on. We never need to enumerate all possible sentences. The decomposition is also exact (no assumption is made), so any approximation lives only in how each conditional is represented.

### Question 2: What independence assumption does X1 → X2 → X3 → X4 make?
Each word is conditionally independent of all earlier words given the immediately preceding word (the Markov property):

P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1}), equivalently X_t ⟂ X_1, ..., X_{t-2} | X_{t-1}.

### Question 3: Conditional probability distributions (first order)
Training data: 6 sentences, so "the" is followed 12 times (cat 3, dog 3, mat 2, rug 2, park 2).

| Current word | Distribution P(next \| current) |
|---|---|
| `<START>` | the = 1.000 |
| the | cat = 0.250, dog = 0.250, mat = 0.167, rug = 0.167, park = 0.167 |
| cat | sat = 0.667, ran = 0.333 |
| dog | sat = 0.667, ran = 0.333 |
| sat | on = 1.000 |
| ran | to = 1.000 |
| on | the = 1.000 |
| to | the = 1.000 |
| mat / rug / park | `<END>` = 1.000 |

Zero-probability transitions (never observed) for the required words:
- **the**: `<END>`, on, ran, sat, the, to
- **cat / dog**: everything except sat and ran (e.g. cat→cat, cat→mat, cat→`<END>`)
- **sat**: everything except on
- **ran**: everything except to

Full lists are in `results/results.txt`. Note that a zero probability means "not observed in 6 sentences", not "impossible in English".

### Question 4: Where are the transition counts stored?
In `FirstOrderLM.counts`, a nested dictionary `counts[prev][next]`, filled in `fit()` by looping over consecutive token pairs (`zip(tokens[:-1], tokens[1:])`).

### Question 5: Where is P(X_t | X_{t-1}) computed?
Also in `fit()`: for each `prev`, `total = sum(counts[prev].values())` and `probs[prev][w] = count / total`. The result is stored in `self.probs` and read via `distribution()`.

### Question 6: How does the program choose the next word?
There are two modes. `sample_next()` **samples** from the distribution using `random.choices(words, weights=probs)`, while `most_probable()` is **greedy** and always returns the argmax. Greedy is deterministic: the same context always gives the same word, so the whole sentence is fixed. Sampling picks each word with its model probability, so a word with P = 0.25 is chosen a quarter of the time. Sampling reproduces the diversity of the training distribution; greedy collapses it onto the single most likely path.

### Question 7: What happens with a word that has no observed transition?
`distribution()` returns `{}`, `most_probable()` and `sample_next()` return `None`, and `generate()` stops the sentence (it treats `None` like `<END>`). In the tests, `distribution('bird')` gives `{}`. Without this guard the program would raise a `KeyError` or `IndexError`. A more sophisticated fix is smoothing (e.g. add-one), which gives unseen transitions a small non-zero probability.

### Question 8: A total of 0.87 instead of 1.0
It means the implementation does not define a valid probability distribution for that context: 13% of the probability mass is missing. Likely causes are a wrong denominator (e.g. counting only some of the successors, or a count that was not included in the total), counts that were dropped or double-normalised, or probabilities that were rounded or filtered (e.g. removing low-probability words after normalising). Sampling from such a table would be biased.

Our tests: every context in both models sums to 1.000000 (see `results/results.txt`).

### Question 9: Do the most probable predictions match human expectation?
Mostly, but not always. `argmax P(. | the) = cat`, but cat and dog are tied at 0.25, and the choice of cat is only my deterministic tie-break (alphabetical); a human would say "the" can be followed by almost any noun. The model only knows the six training sentences, so it says nothing about, e.g., "the bird" (probability 0). It is also a statement about **frequency in this corpus**, not about meaning or grammar. A probability model captures the statistics of its data; human expectation draws on world knowledge and a far larger experience of language.

### Question 10: Greedy vs sampling
Sampling gives much more variation. Greedy produced one sentence, repeated five times: `the cat sat on the cat sat on the cat ...` (it runs to the max length of 30 words because the greedy path loops `the → cat → sat → on → the` and never reaches `<END>`). Five sampled sentences gave 3 distinct outputs (e.g. `the park`, `the rug`, `the dog sat on the rug`). Greedy has no randomness; every step takes the max, so all runs are identical. Greedy can also get trapped in high-probability loops, which is a known weakness of greedy decoding.

Note also that the first-order sampled sentences are often bad (`the park`, `the mat`): the model cannot remember that "the" followed by "park" only makes sense after "ran to".

### Question 11: Second order vs first order
1. **Graph structure**: each X_t has two parents (X_{t-2} and X_{t-1}) instead of one, so the chain gets extra edges X_{t-2} → X_t.
2. **CPT**: the table is indexed by pairs (w_{t-2}, w_{t-1}), so up to |V|² rows instead of |V|. Here 121 possible contexts vs 11.
3. **Context**: the model sees two previous words, so it can distinguish "sat on **the**" from "ran to **the**": `P(next | on, the)` = mat/rug, `P(next | to, the)` = park.
4. **Data needed**: many more parameters need estimating, so far more data is needed for the same reliability; most contexts are never observed (106 of 121 possible contexts here).

### Question 12: Why more context helps, and why it makes estimation harder
More context removes ambiguity: with only "the" as context, the next word could be cat, dog, mat, rug or park. With two words, "on the" points to mat/rug and "to the" to park. So the conditional distribution is sharper and closer to the true P(X_t | X_1..X_{t-1}). But the CPT grows exponentially with the order: |V|^k contexts (each with |V| entries), i.e. |V|^(k+1) entries in total. Here: 121 (first) vs 1331 (second) possible entries, but only 17 vs 19 are non-zero. Because the data is fixed, each context is seen fewer times (sparsity), estimates get noisy, and unseen contexts have no prediction at all. This is the bias-variance trade-off: low order = high bias, low variance; high order = low bias, high variance.

### Question 13: Why is Approach B preferable?
- **Specifying intended behaviour**: B defines what is computed (the conditional P(X_t | X_{t-1}), from counts, with sampling), so the LLM cannot silently substitute something else, e.g. a pretrained model.
- **Understanding the representation**: B forces me to know that the model is a CPT of counts, so I can read the code and find where each part lives.
- **Validating the implementation**: with a specification I can check the code against it. With A there is nothing to check against.
- **Testing probabilistic invariants**: B gives testable properties: rows sum to 1, probabilities are in [0, 1], and frequencies match counts.
- **Implementation vs model**: the model (a Bayesian network with a given factorisation) is separate from the program that implements it. A program can run and produce plausible text and still not implement the intended model. B lets me tell the difference.

### Question 14: What did the Bayesian-network view add?
- **Representation of dependencies**: the graph X_{t-1} → X_t makes it explicit what each word depends on. Moving to the second-order model was just adding an edge.
- **Factorisation of the joint**: P(X1..XT) = ∏ P(X_t | parents(X_t)) told me exactly which tables to build and how to compute the probability of a sentence.
- **Principled generation**: ancestral sampling (sample parents first, then children) is exactly the sample-append-sample loop.
- **Reasoning about independence assumptions**: the graph shows X_t ⟂ X_{1..t-2} | X_{t-1}, and shows what is lost, e.g. `the cat sat on the cat sat...` in first-order output.
- **Effect of increasing context**: parent-set size determines CPT size, which explains the data-hunger of the second-order model.
- **Testing against the specification**: each CPT row must be a distribution, which gave the normalisation test.

---

## Results summary (from `results/results.txt`)

| | First order | Second order |
|---|---|---|
| Vocabulary (incl. START/END) | 12 | 12 |
| Observed contexts | 11 | 15 |
| Non-zero parameters | 17 | 19 |
| Possible contexts | 11 | 121 |
| Contexts never observed | 0 | 106 |
| Full CPT size (contexts × next tokens) | 121 | 1331 |
| Distinct sentences in 1000 samples | 175 | 6 |
| Novel sentences (not in training) among those | 169 | 0 |
| Sampled sentences equal to a training sentence | 14.5% | 100% |
| Greedy output | loops forever (`the cat sat on the cat ...`) | `the cat sat on the mat` |

Qualitative comparison:
- First order, e.g. `the cat sat on the dog ran to the cat sat on the ...` and `the park`. It is diverse but often incoherent, because a single word of context lets it jump between sentence templates.
- Second order, e.g. `the dog ran to the park`, `the cat sat on the rug`. It is fully coherent, but on this tiny corpus it only reproduces (and slightly recombines) the training sentences: 6 distinct sentences, 0 novel. It has memorised the data (overfitting), which is the flip side of Question 12.

---

## Reflection: how the LLM was used and how the output was validated

**How the LLM was used.** I did not ask the LLM to "write a language model". I first described the probabilistic model: variables are tokens X_t, the dependency is X_{t-1} → X_t, the quantity to estimate is P(X_t | X_{t-1}) from transition counts, and generation is repeated sampling until `<END>`. I also said no machine-learning libraries or pretrained models were allowed. For the second-order model I stated what had to change before asking for code: the context becomes the pair (X_{t-2}, X_{t-1}), the counts are over triples, and everything else stays the same. The LLM produced the code, and I treated it as a draft to check against that specification.

**How I inspected it.** I traced each requirement to a place in the code: counts are stored in `counts[prev][next]`, probabilities are computed in `fit()` by dividing each count by the row total, and the next word is chosen by `random.choices` (sampling) or by taking the argmax (greedy). I confirmed that no library other than the standard library is used.

**Examples of generated code I inspected and checked:**
1. *Tie-breaking in `most_probable`.* `P(cat | the) = P(dog | the) = 0.25`, so a plain `max(dist, key=dist.get)` would return whichever word was inserted first, and the "most probable" word would change if the order of the training sentences changed. The code sorts by `(-probability, token)`, so ties are broken alphabetically and the result is reproducible. I checked this by hand: the answer for "the" is "cat" only because of the tie-break, and I report that in Question 9.
2. *Unseen contexts and greedy loops.* For a word never seen in training, `distribution()` returns `{}` and generation stops instead of raising a `KeyError` (tested with `bird`). Greedy generation follows the loop `the → cat → sat → on → the → cat ...` and never reaches `<END>`, so the `max_len` cap is necessary. Without it the program would not terminate. I found this by running greedy mode and seeing the repeated output.

**How I validated it.**
- *Normalisation test:* for every context in both models, the probabilities sum to 1.0 (all rows show 1.000000). A row like 0.87 would show a wrong denominator or lost probability mass.
- *Hand check against counts:* I counted the training data by hand, for example "the" is followed 12 times (cat 3, dog 3, mat 2, rug 2, park 2), and compared with the printed table (0.250, 0.250, 0.167, 0.167, 0.167).
- *Support check:* every generated transition has a non-zero count in the training data, so the model never invents an unobserved transition.
- *Behavioural check:* sampling gives varied output and greedy gives a single repeated output, as the model predicts.

**What I learned.** The code running and printing plausible sentences is not evidence that it implements the intended model. The specification, the invariants (rows sum to 1) and hand-computed values are what show that it does. Writing the model down first as a Bayesian network (nodes, parents, factorisation) is what made those checks possible.
