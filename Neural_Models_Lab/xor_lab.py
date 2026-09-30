"""Neural-models lab: XOR, backpropagation checks, symmetry, activations, 3-class softmax.

Model (baseline): 2 inputs -> 2 hidden units -> 1 output logit
Loss            : BCEWithLogitsLoss (sigmoid + binary cross-entropy in one stable op)
Optimiser       : Adam, full batch (all four examples every step)
"""
import math
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)      # float64 so finite-difference checks are clean

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

ACTS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}

log = []
def out(s=""):
    print(s)
    log.append(s)


def make_model(act="sigmoid", seed=0, zero_init=False):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 2), ACTS[act](), nn.Linear(2, 1))
    if zero_init:
        for p in net.parameters():
            nn.init.zeros_(p)
    return net


def train(net, steps=3000, lr=0.05, record_every=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    hist = []
    first_grad_norm = None
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(X), Y)             # forward pass + scalar loss
        loss.backward()                       # reverse-mode AD (backpropagation)
        if step == 0:
            first_grad_norm = net[0].weight.grad.norm().item()
            first_loss = loss.item()
        if record_every and step % record_every == 0:
            hist.append((step, loss.item(), net[0].weight.detach().clone()))
        opt.step()                            # parameter update
    with torch.no_grad():
        logits = net(X)
        final_loss = loss_fn(logits, Y).item()
        probs = torch.sigmoid(logits).squeeze()
        labels = (probs > 0.5).long()
    correct = bool((labels == Y.squeeze().long()).all())
    return dict(first_loss=first_loss, final_loss=final_loss, probs=probs, labels=labels,
                correct=correct, first_grad_norm=first_grad_norm, hist=hist)


# ----------------------------------------------------------------------------
out("=== Task 1: what a purely affine model can do ===")
torch.manual_seed(0)
linear = nn.Linear(2, 1)                    # one affine map + sigmoid (inside the loss)
opt = torch.optim.Adam(linear.parameters(), lr=0.05)
lf = nn.BCEWithLogitsLoss()
for _ in range(5000):
    opt.zero_grad(); l = lf(linear(X), Y); l.backward(); opt.step()
with torch.no_grad():
    p = torch.sigmoid(linear(X)).squeeze()
out(f"Affine + sigmoid after 5000 steps: loss={lf(linear(X), Y).item():.4f}, "
    f"probabilities={[round(v, 3) for v in p.tolist()]}, correct={int(((p > 0.5).long() == Y.squeeze().long()).sum())}/4")
out("(ln 2 = 0.6931 is the loss of predicting 0.5 everywhere.)")

# ----------------------------------------------------------------------------
out("\n=== Task 4A: basic learning check (2-2-1, sigmoid hidden, seed 0) ===")
net = make_model("sigmoid", seed=0)
r = train(net)
out(f"initial loss = {r['first_loss']:.4f}, final loss = {r['final_loss']:.6f}")
for xi, pi, li in zip(X.tolist(), r["probs"].tolist(), r["labels"].tolist()):
    out(f"  input {tuple(int(v) for v in xi)} -> p={pi:.4f}, label={li}")
out(f"all four correct: {r['correct']}")

# ----------------------------------------------------------------------------
out("\n=== Task 4B: backpropagation check (fresh sigmoid network, seed 0, before training) ===")
net = make_model("sigmoid", seed=0)
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(net(X), Y)
loss.backward()
W1_grad = net[0].weight.grad.clone()
out(f"autograd dL/dW1 =\n{W1_grad}")

# (1) manual chain rule:  dL/dz = (sigmoid(z) - y)/N ;  dL/da = dL/dz * W2 * f'(a)
with torch.no_grad():
    a1 = X @ net[0].weight.T + net[0].bias
    h = torch.sigmoid(a1)
    z = h @ net[2].weight.T + net[2].bias
    dz = (torch.sigmoid(z) - Y) / 4
    da = (dz @ net[2].weight) * h * (1 - h)
    manual = da.T @ X
out(f"manual chain rule dL/dW1 =\n{manual}")
out(f"max |autograd - manual| = {(W1_grad - manual).abs().max().item():.2e}")

# (2) finite differences
eps = 1e-6
fd = torch.zeros_like(W1_grad)
for i in range(2):
    for j in range(2):
        with torch.no_grad():
            net[0].weight[i, j] += eps; lp = loss_fn(net(X), Y).item()
            net[0].weight[i, j] -= 2 * eps; lm = loss_fn(net(X), Y).item()
            net[0].weight[i, j] += eps
        fd[i, j] = (lp - lm) / (2 * eps)
out(f"max |autograd - finite difference| = {(W1_grad - fd).abs().max().item():.2e}")

# (3) mean loss => average of per-example gradients
per_ex = []
for k in range(4):
    net.zero_grad()
    loss_fn(net(X[k:k + 1]), Y[k:k + 1]).backward()
    per_ex.append(net[0].weight.grad.clone())
avg = torch.stack(per_ex).mean(0)
out(f"max |mean-loss grad - average of 4 per-example grads| = {(W1_grad - avg).abs().max().item():.2e}")

# ----------------------------------------------------------------------------
out("\n=== Task 4C: symmetry experiment (all weights and biases = 0, sigmoid hidden) ===")
net0 = make_model("sigmoid", zero_init=True)
r0 = train(net0, steps=3000, record_every=500)
for step, lval, W in r0["hist"]:
    out(f"step {step:4d}: loss={lval:.6f}  W1 row0={W[0].tolist()}  row1={W[1].tolist()}  rows identical={bool(torch.equal(W[0], W[1]))}")
out(f"final loss={r0['final_loss']:.6f}, probabilities={[round(v, 4) for v in r0['probs'].tolist()]}, "
    f"all four correct={r0['correct']}")
out(f"final rows identical: {bool(torch.equal(net0[0].weight[0], net0[0].weight[1]))}")

# zero init is a degenerate case (see README): W2 = 0 makes dL/dW1 = 0 and the balanced XOR
# targets make dL/dW2 = 0 as well, so nothing ever moves. A non-zero *identical* start shows the
# symmetry problem while gradients are non-zero.
out("\n=== Task 4C (second variant): every weight = 0.5, every bias = 0 (identical but non-zero) ===")
netc = make_model("sigmoid", seed=0)
with torch.no_grad():
    for p_ in netc.parameters():
        p_.fill_(0.5 if p_.dim() == 2 else 0.0)
rc = train(netc, steps=3000, record_every=500)
for step, lval, W in rc["hist"]:
    out(f"step {step:4d}: loss={lval:.6f}  W1 row0={[round(v, 4) for v in W[0].tolist()]}  row1={[round(v, 4) for v in W[1].tolist()]}  "
        f"identical={bool(torch.allclose(W[0], W[1], atol=1e-12))}")
out(f"final loss={rc['final_loss']:.6f}, probabilities={[round(v, 4) for v in rc['probs'].tolist()]}, all four correct={rc['correct']}")
out(f"final rows identical: {bool(torch.allclose(netc[0].weight[0], netc[0].weight[1], atol=1e-12))}; "
    f"W2 = {netc[2].weight.tolist()}")

# ----------------------------------------------------------------------------
out("\n=== Task 4D: activation experiment (random init, seed 0, Adam lr=0.05, 3000 steps) ===")
out("| Hidden activation | Final loss | 4/4 correct? | Early ||grad W1||_2 (step 0) |")
out("|---|---|---|---|")
rows = {}
for act in ACTS:
    n = make_model(act, seed=0)
    rr = train(n)
    rows[act] = rr
    out(f"| {act} | {rr['final_loss']:.6f} | {rr['correct']} | {rr['first_grad_norm']:.4f} |")

out("\nRobustness: 30 random seeds per activation (same settings)")
out("| Hidden activation | seeds reaching 4/4 | median final loss | stuck at loss ~0.3466 (two examples fit, two left at p=0.5) |")
out("|---|---|---|---|")
for act in ACTS:
    res = [train(make_model(act, seed=s)) for s in range(30)]
    ok = sum(x["correct"] for x in res)
    med = sorted(x["final_loss"] for x in res)[15]
    stuck = sum(abs(x["final_loss"] - 0.34657) < 0.01 for x in res)
    out(f"| {act} | {ok}/30 | {med:.6f} | {stuck} |")

# saturation vs dead-ReLU diagnostics on the trained seed-0 networks
out("\nDiagnostics on the trained seed-0 networks (pre-activations a and activations h per input):")
for act in ACTS:
    n = make_model(act, seed=0); train(n)
    with torch.no_grad():
        a = X @ n[0].weight.T + n[0].bias
        h = ACTS[act]()(a)
    out(f"  {act}: pre-activation range [{a.min().item():.2f}, {a.max().item():.2f}]; "
        f"activation range [{h.min().item():.3f}, {h.max().item():.3f}]")
    if act == "sigmoid":
        d = (h * (1 - h))
        out(f"     sigmoid derivative h(1-h): min {d.min().item():.4f}, max {d.max().item():.4f}  (near 0 => saturated)")
    if act == "relu":
        out(f"     fraction of (example, unit) pairs with a<=0 (derivative 0): {(a <= 0).double().mean().item():.2f}")

# ----------------------------------------------------------------------------
out("\n=== Task 5: three-class extension (0: (0,0), 1: (0,1)/(1,0), 2: (1,1)) ===")
Y3 = torch.tensor([0, 1, 1, 2])
torch.manual_seed(0)
net3 = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 3))   # only the output/loss part changed
out(f"Predicted shapes: final weight matrix {tuple(net3[2].weight.shape)} (3 x 2), 3 logits per example")
ce = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net3.parameters(), lr=0.05)
for step in range(2000):
    opt.zero_grad(); l = ce(net3(X), Y3); l.backward(); opt.step()
with torch.no_grad():
    logits = net3(X)
    P = torch.softmax(logits, dim=1)
out(f"final cross-entropy loss = {ce(logits, Y3).item():.6f}")
for xi, pi in zip(X.tolist(), P.tolist()):
    out(f"  input {tuple(int(v) for v in xi)} -> p = {[round(v, 4) for v in pi]}, argmax = {int(torch.tensor(pi).argmax())}")
out(f"predictions = {P.argmax(1).tolist()}, targets = {Y3.tolist()}, all correct = {bool((P.argmax(1) == Y3).all())}")
out(f"softmax vector for input (0,1) = {[round(v, 6) for v in P[1].tolist()]}, sum = {P[1].sum().item():.12f}")

# gradient wrt logits is p - y (for the SUM over examples; the mean divides by N)
lg = net3(X).detach().requires_grad_(True)
nn.CrossEntropyLoss(reduction="sum")(lg, Y3).backward()
onehot = torch.nn.functional.one_hot(Y3, 3).double()
out(f"max |autograd dL/dlogits - (p - y)| (sum reduction) = {(lg.grad - (torch.softmax(lg.detach(), 1) - onehot)).abs().max().item():.2e}")
lg2 = net3(X).detach().requires_grad_(True)
nn.CrossEntropyLoss()(lg2, Y3).backward()
out(f"with mean reduction the gradient is (p - y)/4: max error = {(lg2.grad - (torch.softmax(lg2.detach(), 1) - onehot) / 4).abs().max().item():.2e}")

# shift invariance and numerical stability
z = logits[1]
p_plain = torch.softmax(z, 0)
p_shift = torch.softmax(z + 100, 0)
out(f"\nAdding 100 to all logits: max |p_shift - p| = {(p_plain - p_shift).abs().max().item():.2e}")
big = z + 1000
naive = torch.exp(big) / torch.exp(big).sum()
stable = torch.exp(big - big.max()) / torch.exp(big - big.max()).sum()
out(f"Naive softmax with logits + 1000: {naive.tolist()}  (exp overflows to inf, inf/inf = nan)")
out(f"Max-subtracted softmax with logits + 1000: {[round(v, 6) for v in stable.tolist()]}")

with open("results/results.txt", "w") as f:
    f.write("\n".join(log) + "\n")
