# Neural Models Lab: Learning, Depth, Activations and Output Layers

Requires Python 3, NumPy and PyTorch (CPU is enough; tested with torch 2.14).

```bash
pip install torch numpy
python xor_lab.py        # ~1 min on CPU; writes results/results.txt
```

| File | Purpose |
|---|---|
| `xor_lab.py` | Everything: affine baseline, 2-2-1 XOR network, gradient checks, symmetry experiments, activation comparison (30 seeds), 3-class extension |
| `results/results.txt` | Full output of the run (all numbers below come from it) |

---

## Task 1: Understand the problem before coding

**Specification.** Input space 𝒳 = {0,1}², output space 𝒴 = {0,1}. Examples: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0. The device must raise a warning exactly when the two sensors disagree (XOR).

**Sketch.** In the (x₁, x₂) plane, class 0 sits at opposite corners (0,0) and (1,1), and class 1 at the other two corners (0,1) and (1,0).

**Why no single straight boundary works.** Any line that puts (0,1) and (1,0) on one side has their midpoint (0.5, 0.5) on that side too (a half-plane is convex); but (0.5, 0.5) is also the midpoint of (0,0) and (1,1), which must be on the other side. Contradiction.

**Prediction for one affine transformation + sigmoid:** it cannot learn XOR; the best it can do is output ≈ 0.5 for every input (loss ln 2 ≈ 0.693) and get at most 3 of 4 right.
**Confirmed in the run:** after 5000 Adam steps the loss was 0.6931 and all four probabilities were 0.5.

**Think About It.** XOR lets us test the claim that a model needs the right *kind* of representation, not just many parameters: any stack of purely affine layers collapses to one affine map, so it fails however many parameters it has, whereas a nonlinear hidden layer makes the classes separable in the hidden space.

## Task 2: Model design and validation criteria

**Model:** 2 inputs → 2 hidden units (nonlinearity f) → 1 output logit; ŷ = σ(W²f(W¹x + b¹) + b²). **Loss:** binary cross-entropy (implemented as `BCEWithLogitsLoss` on the logit). **Optimiser:** Adam, full batch of all 4 examples, lr = 0.05, 3000 steps. Hidden activations compared: sigmoid, tanh, ReLU.

1. **Why is the hidden nonlinearity necessary?** Without it the network is an affine map of the input and can only draw one linear boundary (Task 1). The nonlinear hidden layer re-represents the inputs so that the two classes become linearly separable at the output; this is a claim about *representation*, not about the number of parameters.
2. **Why sigmoid + binary cross-entropy?** The target is one yes/no answer, so the output should be a probability in (0,1) (sigmoid) and the loss should be the negative log-likelihood of a Bernoulli variable (BCE). The pair also gives the simple logit gradient p − y and avoids the vanishing gradient that sigmoid + squared error would suffer when the output saturates.
3. **Evidence of successful learning (validation criteria):**
   - final loss well below ln 2 = 0.693 (here < 0.001);
   - all four thresholded predictions correct;
   - non-zero gradient for the first-layer weights, and agreement of autograd with a manual chain-rule computation and a finite-difference check;
   - repeated-run behaviour over many random seeds (the result is not a single lucky seed).

**Think About It: who determines what each hidden unit computes?** Nothing gives the hidden units targets. The *gradient signal* does: backpropagation sends the output error back through W² (∂L/∂a = (W²)ᵀδ ⊙ f′(a)), so each hidden unit is pushed to compute whatever feature reduces the output loss. What a hidden unit represents is a by-product of this learning signal, together with the random initial weights.

## Task 3: LLM-generated implementation

**Prompt used**

```
Generate minimal PyTorch code for the following model and dataset. Do not change the
architecture or task.
Data: the four XOR examples (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
Model: 2 inputs -> 2 hidden units (sigmoid) -> 1 output logit; random weight
initialisation. Loss: BCEWithLogitsLoss. Full-batch training for a few thousand
lightweight CPU steps. After training, report the final loss, all four probabilities,
thresholded labels, and one parameter-gradient tensor after backward(). Set a random
seed for reproducibility and explain each test in one sentence.
```

**Code inspection.** Forward pass: `net(X)` inside `train()`. Scalar loss: `loss_fn(net(X), Y)`. Reverse-mode AD: `loss.backward()`. Parameter update: `opt.step()`.

**Changes made / engineering settings:** (1) `float64` as the default dtype so the finite-difference check is clean; (2) Adam with lr = 0.05 and 3000 steps chosen because plain small-step SGD is slow on this tiny problem; the task (architecture, data, loss) was not changed.

**Think About It: verifiable by reading vs requiring execution.** Reading suffices for: the architecture (2-2-1), that the data are the four XOR examples, that BCE-with-logits is the loss, where `backward()` and the optimiser step are, and that the seed is set. Only running the code shows: whether the loss decreases, whether the labels are right, the gradient values, and how often the training ends in a bad minimum.

## Task 4: Execution and diagnosis

### A: Basic learning check (sigmoid, seed 0)
Initial loss 0.7072 → final loss **0.000284**. Probabilities: (0,0)→0.0003, (0,1)→0.9997, (1,0)→0.9997, (1,1)→0.0003; thresholded labels 0,1,1,0 → **4/4 correct**.

### B: Backpropagation check
`net[0].weight.grad` holds ∂L/∂W¹ (a 2×2 matrix: entry (i,j) is the derivative of the loss with respect to the weight from input j to hidden unit i). On the untrained seed-0 network it was `[[0.0054, 0.0071], [0.0031, 0.0017]]`, and:
- max |autograd − manual chain rule| = 8.7e-19,
- max |autograd − finite differences| = 4.2e-11,
- max |gradient of the mean loss − average of the 4 per-example gradients| = 0.

The last check holds because the loss is L = (1/4) Σₙ Lₙ and differentiation is linear, so ∂L/∂W = (1/4) Σₙ ∂Lₙ/∂W.

### C: Symmetry experiment
- **All weights and biases = 0:** the loss stayed exactly 0.693147 for 3000 steps, W¹ never moved (both rows `[0, 0]`), and all predictions were 0.5. This is a degenerate version of the symmetry problem: with W² = 0 the hidden gradients ∂L/∂W¹ are exactly zero, and with balanced XOR targets ∂L/∂W² (and the output bias gradient) are also exactly zero at 0.5-predictions, so nothing ever changes.
- **Identical but non-zero start (every weight 0.5, biases 0):** here the gradients are non-zero. The two rows of W¹ stayed **identical at every recorded step** (checked to 1e-12), W² ended with two equal entries `[0.2244, 0.2244]`, the loss ended at 0.693147 and all four predictions were 0.5 (not solved).
- **Explanation:** two hidden units with identical incoming weights compute the same output, so they receive the same gradient (∂L/∂a for unit i depends only on its own activation, the same W² value and the same input), and the update keeps them identical forever. The network then behaves like one hidden unit, which is (a monotone function of) an affine function of the input, and cannot represent XOR. Random initialisation *breaks the symmetry*.

### D: Activation experiment (random init, seed 0, Adam lr 0.05, 3000 steps)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W¹ L‖₂ (step 0) |
|---|---|---|---|
| Sigmoid | 0.000284 | Yes | 0.0096 |
| Tanh | 0.000108 | Yes | 0.0437 |
| ReLU | 0.693147 | **No** | 0.0412 |

**Repeated-run behaviour (30 seeds, same settings)**

| Hidden activation | Seeds reaching 4/4 | Median final loss | Runs stuck at loss ≈ 0.3466 |
|---|---|---|---|
| Sigmoid | 12/30 | 0.3467 | 11 |
| Tanh | 16/30 | 0.000157 | 13 |
| ReLU | 9/30 | 0.4774 | 2 |

**Interpretation (for this experiment only, four data points and 2 hidden units).**
- *Engineering observation:* the step-0 gradient norm was about 4.5 times smaller for sigmoid (0.0096) than for tanh (0.0437). A sigmoid unit's derivative never exceeds 0.25, while tanh's reaches 1, so gradients through sigmoid units are scaled down by up to 4 at each layer. Even so, the sigmoid run trained successfully on seed 0, because Adam normalises the step size; the raw gradient scale is not the same as training speed.
- *Scientific explanation of failures:* many runs (11 sigmoid, 13 tanh, 2 ReLU of 30) ended in a partial solution with loss ≈ 0.3466 = ln 2 / 2. Inspecting them shows two examples fitted almost perfectly (p ≈ 0 or 1) and the other two left at exactly p = 0.5, i.e. the two hidden units have found only part of the XOR structure. With only 2 hidden units and small random weights, this is a common outcome. With ReLU on seed 0 the trained network's pre-activations on all four inputs were in [0.05, 0.39], i.e. **positive everywhere**, so both units acted as the identity and the network collapsed to an affine map (the Task 1 situation): loss ln 2, predictions 0.5. This is not a "dead ReLU" (zero activation); it is an "always-on ReLU" that never uses its nonlinearity.
- I do **not** conclude that any activation is universally best: tanh had the highest success rate here (16/30), but with 30 seeds and this tiny problem the differences between 12, 16 and 9 successes are not strong evidence.

**Think About It: saturation vs. dead ReLU.** Inspect the pre-activations a and activations h. A *saturated sigmoid* unit has large |a| (e.g. −15 or +17 in the trained seed-0 network) with h close to 0 or 1 and h(1−h) ≈ 0 (maximum 0.0077 there). A *dead ReLU* has a ≤ 0 for all inputs, so h = 0 and the derivative is exactly 0. In both cases the gradient is small, but the diagnostic values differ.

## Task 5: Three-class extension

Classes: 0 = (0,0), 1 = (0,1) or (1,0), 2 = (1,1). Only the output/loss was changed: 2 → 2 (tanh) → **3 logits**, `CrossEntropyLoss`.

**Predictions made before running**
1. Final weight matrix: 3 × 2 (3 output logits, 2 hidden units); PyTorch reports `weight.shape = (3, 2)`.
2. Logits per example: 3.
3. Softmax probabilities sum to one because pᵢ = eᶻⁱ / Σⱼ eᶻʲ: the numerators are divided by their own sum.
4. The logit gradient is p − y: for L = −log p_c with p = softmax(z), ∂L/∂zᵢ = pᵢ − 𝟙[i = c] (the softmax Jacobian and the log combine so that the exponentials cancel). With a mean over N examples the gradient is (p − y)/N.

**Results.** Final loss 0.000069; predicted classes [0, 1, 1, 2] = targets. Probability vectors: (0,0) → [0.9999, 0.0001, 0.0000]; (0,1) → [0.0000, 0.9999, 0.0000]; (1,0) → same as (0,1); (1,1) → [0.0000, 0.0001, 0.9999]. For input (0,1), the softmax vector sums to 1.000000000000. Numerical check: max |autograd − (p − y)| = 6.8e-20 (sum reduction) and the mean-reduction gradient equals (p − y)/4 to 1.7e-20.

**Optional diagnostic.** Adding 100 to all logits changed the probabilities by at most 2.1e-19 (round-off). Softmax is shift-invariant because the common factor e¹⁰⁰ cancels between numerator and denominator. A naive implementation with logits + 1000 gives `nan` (exp overflows to ∞, ∞/∞); subtracting the maximum logit first gives the correct result. Stable implementations therefore subtract the max before exponentiating.

**Think About It: language-model next-token prediction.** Mathematically unchanged when the classes become tens of thousands of tokens: logits, softmax, cross-entropy, the p − y gradient, and the need for a numerically stable softmax. What changes dramatically is the surrounding architecture: instead of a 2-unit hidden layer, a deep network (embeddings, attention, many layers) computes the logits from a long context; the output matrix becomes huge (vocab × hidden size), and the cost of the softmax and memory/training scale accordingly.

## Reflection questions

1. **Depth vs nonlinearity.** Adding affine-only layers does not help: the composition is still one affine map, so depth without nonlinearity cannot represent XOR. The nonlinear hidden layer is what makes XOR representable.
2. **Evidence that backprop gave a *useful* signal.** Beyond a non-zero gradient, the loss fell from 0.707 to 0.0003, all four predictions became correct, the gradients matched the manual chain rule and finite differences, and the same code with zero/identical weights (where the signal cannot break symmetry) stayed at ln 2. A non-zero gradient alone does not prove learning: the identical-weights run also had non-zero gradients but did not learn.
3. **Why did identical/zero weights prevent distinct features?** Identical incoming weights → identical activations → identical gradients → identical updates, so the units stay identical (and with all-zero weights the gradients even vanish). Random initialisation breaks the symmetry.
4. **Effect of the activation on the gradient.** *Engineering observation:* the step-0 first-layer gradient norm was 0.0096 (sigmoid), 0.0437 (tanh), 0.0412 (ReLU). *Scientific explanation:* sigmoid′ ≤ 0.25, tanh′ ≤ 1, active ReLU′ = 1, so the product of derivative factors in the chain rule is smaller for sigmoid; saturation and inactivity shrink it further. The observed norms are consistent with this but are one seed, not proof.
5. **Why choose the output layer and loss together?** They must match the task and each other: sigmoid + BCE for one binary output, softmax + cross-entropy for one-of-K. The pairing gives the meaningful probabilistic interpretation and the clean gradient p − y, and the fused, stable implementations (`BCEWithLogitsLoss`, `CrossEntropyLoss`) avoid overflow.
6. **LLM: productivity vs verification.** *Productivity:* it produced the training loop, the per-example-gradient and finite-difference check code and the tables quickly. *Verification was essential:* the zero-initialisation experiment did not show the expected "identical rows that keep changing" — the weights never moved at all, which is a different (degenerate) phenomenon that a human must notice and explain, and it needed the extra identical-non-zero-start run; likewise the ReLU seed-0 failure had to be diagnosed by inspecting pre-activations.
7. **Tests to keep when scaling up.** Keep: loss-decrease and overfit-a-tiny-batch tests, output shape checks, softmax/probability normalisation, gradient-norm monitoring per layer, symmetry/initialisation checks, and multi-seed repeatability (on a subset). Too expensive at scale: exhaustive finite-difference gradient checks (they need about two extra forward passes per parameter, i.e. millions for a large model; they are only feasible on a few randomly sampled parameters), and 30-seed sweeps of full training.

## Responsible use of LLMs
The code was generated from a written specification and then verified by running it: autograd against manual and numerical gradients, the symmetry runs, a 30-seed robustness sweep and numerical checks of the softmax gradient. Where the results disagreed with a naive expectation (zero initialisation, ReLU on seed 0), the explanation was worked out from the measurements.
