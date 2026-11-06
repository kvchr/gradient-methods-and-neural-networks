"""Part 2, Problem 2: how a Fashion-MNIST classifier depends on the optimizer,
the number of neurons and the number of hidden layers.

The base model follows the TensorFlow tutorial "Basic classification: Classify images of
clothing" (Flatten -> Dense(128, ReLU) -> Dense(10)), trained for 10 epochs with batch size 32,
written here in PyTorch.

Protocol
- The 60,000 training images are split once into 54,000 for training and 6,000 for validation.
- Every setting is trained with 3 seeds. Settings are compared by validation accuracy.
- The 10,000 test images are used once, at the end, for the setting chosen on validation.

Usage: python fashion_mnist_study.py [--data DIR] [--figures-only]
The dataset is downloaded by torchvision into DIR (default: python/data) on the first run (about 30 MB).
Results go to ../outputs (CSV and JSON) and the figures to ../figures.
--figures-only redraws the figures from the saved CSV without training.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
EPOCHS = 10
BATCH = 32
SEEDS = (0, 1, 2)
N_VAL = 6000
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Optimizers and the learning rates tried for each (the middle value is the usual default)
OPTIMIZERS = {
    "SGD":            (lambda p, lr: torch.optim.SGD(p, lr=lr), (0.001, 0.01, 0.1)),
    "SGD + momentum": (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9), (0.001, 0.01, 0.1)),
    "Nesterov":       (lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, nesterov=True), (0.001, 0.01, 0.1)),
    "Adagrad":        (lambda p, lr: torch.optim.Adagrad(p, lr=lr), (0.001, 0.01, 0.1)),
    "RMSprop":        (lambda p, lr: torch.optim.RMSprop(p, lr=lr), (0.0001, 0.001, 0.01)),
    "Adam":           (lambda p, lr: torch.optim.Adam(p, lr=lr), (0.0001, 0.001, 0.01)),
}
NEURONS = (16, 32, 64, 128, 256, 512)       # one hidden layer, Adam, lr 0.001
LAYERS = (1, 2, 3, 5, 10, 15)               # hidden layers of 128 neurons, Adam, lr 0.001


def load_fashion_mnist(root):
    """Return (x_train, y_train, x_test, y_test) as tensors, pixels scaled to [0, 1]."""
    from torchvision.datasets import FashionMNIST
    train = FashionMNIST(root, train=True, download=True)
    test = FashionMNIST(root, train=False, download=True)
    to_x = lambda d: d.data.reshape(len(d.data), -1).float() / 255.0
    return to_x(train), train.targets.clone(), to_x(test), test.targets.clone()


def make_model(hidden_layers=1, neurons=128):
    layers, width = [], 28 * 28
    for _ in range(hidden_layers):
        layers += [nn.Linear(width, neurons), nn.ReLU()]
        width = neurons
    layers.append(nn.Linear(width, 10))
    return nn.Sequential(*layers)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = sum((model(x[i:i + 4096]).argmax(1) == y[i:i + 4096]).sum().item() for i in range(0, len(x), 4096))
    return correct / len(x)


def train_one(data, seed, make_opt, lr, hidden_layers=1, neurons=128):
    x_tr, y_tr, x_val, y_val = data
    torch.manual_seed(seed)
    model = make_model(hidden_layers, neurons)
    opt = make_opt(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x_tr), generator=gen)      # new shuffle every epoch
        xs, ys = x_tr[order], y_tr[order]
        for i in range(0, len(x_tr), BATCH):
            opt.zero_grad()
            loss = loss_fn(model(xs[i:i + BATCH]), ys[i:i + BATCH])
            loss.backward()
            opt.step()
        history.append(accuracy(model, x_val, y_val))
    return model, {"train_acc": accuracy(model, x_tr, y_tr), "val_acc": history[-1], "val_history": history}


def run_setting(data, rows, study, label, make_opt, lr, hidden_layers=1, neurons=128):
    results = []
    for seed in SEEDS:
        t = time.time()
        _, r = train_one(data, seed, make_opt, lr, hidden_layers, neurons)
        rows.append({"study": study, "setting": label, "lr": lr, "hidden_layers": hidden_layers,
                     "neurons": neurons, "seed": seed, "train_acc": r["train_acc"], "val_acc": r["val_acc"],
                     "val_history": json.dumps([round(v, 5) for v in r["val_history"]]),
                     "seconds": round(time.time() - t, 1)})
        results.append(r)
    tr = np.array([r["train_acc"] for r in results])
    va = np.array([r["val_acc"] for r in results])
    print(f"  {study:9s} {label:22s} lr={lr:<7g} train {100 * tr.mean():6.2f}  "
          f"val {100 * va.mean():6.2f} +- {100 * va.std(ddof=1):4.2f}", flush=True)
    return va.mean()


def make_figures(rows, figdir):
    """Two figures: optimizers, and neurons / hidden layers (mean of the seeds, bars = min..max)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8,
                         "legend.frameon": False})

    def stats(study, setting, lr=None):
        sel = [r for r in rows if r["study"] == study and r["setting"] == setting and (lr is None or r["lr"] == lr)]
        tr = np.array([r["train_acc"] for r in sel]) * 100
        va = np.array([r["val_acc"] for r in sel]) * 100
        hist = np.array([json.loads(r["val_history"]) for r in sel]) * 100
        return tr, va, hist

    # Figure 1: optimizers
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for c, (name, (_, lrs)) in zip(series, OPTIMIZERS.items()):
        means, best = [], (-1, None)
        for lr in lrs:
            _, va, hist = stats("optimizer", name, lr)
            means.append(va.mean())
            a1.plot([lr] * len(va), va, "o", color=c, ms=3, alpha=0.5)
            if va.mean() > best[0]:
                best = (va.mean(), hist)
        a1.plot(lrs, means, "-o", color=c, lw=2, ms=6, label=name)
        a2.plot(range(1, EPOCHS + 1), best[1].mean(0), "-", color=c, lw=2, label=name)
    all_val = [100 * r["val_acc"] for r in rows if r["study"] == "optimizer"]
    a1.set_xscale("log")
    a1.set_xlabel("learning rate")
    a1.set_ylabel("validation accuracy (%)")
    a1.set_ylim(np.floor(min(all_val)) - 1, np.ceil(max(all_val)) + 0.5)
    a1.set_title("After 10 epochs (line: mean of 3 seeds, dots: each seed)", fontsize=10)
    a1.legend(loc="lower center", ncol=2, fontsize=9)
    a2.set_xlabel("epoch")
    a2.set_ylabel("validation accuracy (%)")
    a2.set_title("Each optimizer at its best learning rate (mean of 3 seeds)", fontsize=10)
    a2.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figdir / "p2_2_optimizers.png", dpi=200)
    plt.close(fig)

    # Figure 2: neurons and hidden layers
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, study, values, label, xlab in ((axes[0], "neurons", NEURONS, "{} neurons", "neurons in the hidden layer"),
                                           (axes[1], "layers", LAYERS, None, "hidden layers (128 neurons each)")):
        tr_m, va_m, va_lo, va_hi = [], [], [], []
        for v in values:
            setting = label.format(v) if label else layer_label(v)
            tr, va, _ = stats(study, setting)
            tr_m.append(tr.mean()); va_m.append(va.mean()); va_lo.append(va.min()); va_hi.append(va.max())
        va_m = np.array(va_m)
        ax.plot(values, tr_m, "-s", color=series[0], lw=2, ms=6, label="training set")
        ax.errorbar(values, va_m, yerr=[va_m - np.array(va_lo), np.array(va_hi) - va_m], fmt="-o",
                    color=series[1], lw=2, ms=6, capsize=3, label="validation set")
        ax.set_xscale("log", base=2)
        ax.set_xticks(values)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(xlab)
        ax.set_ylabel("accuracy after 10 epochs (%)")
        ax.legend(loc="lower right" if study == "neurons" else "lower left")
    axes[0].set_title("One hidden layer, Adam (lr 0.001)", fontsize=10)
    axes[1].set_title("Adam (lr 0.001)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "p2_2_neurons_layers.png", dpi=200)
    plt.close(fig)


def layer_label(h):
    return f"{h} hidden layer" + ("s" if h > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from the saved CSV, without training")
    args = ap.parse_args()
    out = HERE.parent / "outputs"
    figdir = HERE.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        with open(out / "fashion_mnist_runs.csv", encoding="utf8") as fh:
            rows = [{**r, "lr": float(r["lr"]), "train_acc": float(r["train_acc"]), "val_acc": float(r["val_acc"])}
                    for r in csv.DictReader(fh)]
        make_figures(rows, figdir)
        return

    x_all, y_all, x_test, y_test = load_fashion_mnist(args.data)
    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(1234))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    data = (x_all[tr_idx], y_all[tr_idx], x_all[val_idx], y_all[val_idx])
    print(f"train {len(tr_idx)}, validation {len(val_idx)}, test {len(x_test)} images; "
          f"{EPOCHS} epochs, batch {BATCH}, seeds {SEEDS}", flush=True)

    rows = []
    adam = OPTIMIZERS["Adam"][0]
    best = {}                                   # best validation accuracy per optimizer: (val, lr)
    print("Optimizers (1 hidden layer of 128 neurons):", flush=True)
    for name, (make_opt, lrs) in OPTIMIZERS.items():
        for lr in lrs:
            v = run_setting(data, rows, "optimizer", name, make_opt, lr)
            if v > best.get(name, (-1, None))[0]:
                best[name] = (v, lr)
    print("Neurons (1 hidden layer, Adam, lr 0.001):", flush=True)
    for n in NEURONS:
        run_setting(data, rows, "neurons", f"{n} neurons", adam, 0.001, 1, n)
    print("Hidden layers (128 neurons each, Adam, lr 0.001):", flush=True)
    for h in LAYERS:
        run_setting(data, rows, "layers", layer_label(h), adam, 0.001, h, 128)

    with open(out / "fashion_mnist_runs.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # The setting with the best mean validation accuracy, then its test accuracy (used only here)
    groups = {}
    for r in rows:
        groups.setdefault((r["study"], r["setting"], r["lr"], r["hidden_layers"], r["neurons"]), []).append(r["val_acc"])
    key = max(groups, key=lambda k: np.mean(groups[k]))
    study, label, lr, h, n = key
    make_opt = OPTIMIZERS[label][0] if study == "optimizer" else adam
    tests = []
    for seed in SEEDS:
        model, _ = train_one(data, seed, make_opt, lr, h, n)
        tests.append(accuracy(model, x_test, y_test))
    # Per-class test accuracy of the chosen setting (last seed)
    model.eval()
    with torch.no_grad():
        pred = model(x_test).argmax(1)
    per_class = {CLASSES[c]: (pred[y_test == c] == c).float().mean().item() for c in range(10)}
    summary = {"chosen": {"study": study, "setting": label, "lr": lr, "hidden_layers": h, "neurons": n,
                          "val_acc_mean": float(np.mean(groups[key]))},
               "test_acc": tests, "test_acc_mean": float(np.mean(tests)), "test_acc_std": float(np.std(tests, ddof=1)),
               "per_class_test_acc": per_class,
               "best_lr_per_optimizer": {k: {"lr": v[1], "val_acc_mean": v[0]} for k, v in best.items()}}
    (out / "fashion_mnist_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"Chosen on validation: {label} (lr {lr}, {h} hidden layer(s) of {n}); "
          f"test accuracy {100 * np.mean(tests):.2f} +- {100 * np.std(tests, ddof=1):.2f} %", flush=True)
    make_figures(rows, figdir)


if __name__ == "__main__":
    main()
