"""Full evaluation: accuracy, shortcut check, SNR robustness, calibration, unknown-target detection,
and embedded deployment (ONNX FP32 / INT8, single-core latency).

Usage: python evaluate.py     # needs checkpoints/cnn_seed*.pt and results/baseline_probs.npz
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from config import CLASSES, OOD_CLASS, SNR_SWEEP_DB
from dataset import load
from models import MicroDopplerCNN, n_params
from train import CKPT, predict_logits

RESULTS = Path(__file__).parent / "results"
V_CONTROLLED = 15.0     # m/s: every class has samples in this bulk-velocity range


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(probs, y):
    return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))


def fit_temperature(logits, y):
    grid = np.logspace(-1, 1, 200)
    return grid[np.argmin([nll(softmax(logits / T), y) for T in grid])]


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def ood_metrics(s_in, s_out):
    """Scores: higher = more in-distribution. Returns AUROC and FPR at 95 % TPR."""
    auroc = roc_auc_score(np.r_[np.ones(len(s_in)), np.zeros(len(s_out))], np.r_[s_in, s_out])
    thr = np.percentile(s_in, 5)
    return auroc, float((s_out >= thr).mean())


def logsumexp(z):
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def summarize_probs(probs, data, logits=None):
    """Metrics shared by CNN seeds, ensemble and baseline."""
    te, sw, ood = data["test"], data["snr_sweep"], data["ood"]
    pred = probs["test"].argmax(1)
    ctrl = np.abs(te["v_radial"]) <= V_CONTROLLED
    out = dict(
        acc=float((pred == te["y"]).mean()),
        f1=float(f1_score(te["y"], pred, average="macro")),
        bal_acc_ctrl=float(balanced_accuracy_score(te["y"][ctrl], pred[ctrl])),
        ece=float(ece(probs["test"], te["y"])),
        snr_acc=[float((probs["snr_sweep"][sw["snr"] == s].argmax(1) == sw["y"][sw["snr"] == s]).mean())
                 for s in SNR_SWEEP_DB],
    )
    out["auroc_msp"], out["fpr95_msp"] = ood_metrics(probs["test"].max(1), probs["ood"].max(1))
    if logits is not None:
        out["auroc_energy"], out["fpr95_energy"] = ood_metrics(logsumexp(logits["test"]), logsumexp(logits["ood"]))
    return out


def latency_ms(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times)), float(np.percentile(times, 99))


def deployment(model, data):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    x1 = torch.from_numpy(data["test"]["X"][:1])[:, None]
    fp32_path, int8_path = CKPT / "cnn_fp32.onnx", CKPT / "cnn_int8.onnx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, x1, fp32_path, dynamo=False, input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)

    class Calib(CalibrationDataReader):
        def __init__(self, X):
            self.it = iter(X[:, None, None])        # 300 single-sample batches from the val split

        def get_next(self):
            x = next(self.it, None)                 # (1, 1, freq, time)
            return None if x is None else {"x": x}

    prep_path = CKPT / "cnn_fp32_prep.onnx"
    quant_pre_process(str(fp32_path), str(prep_path))
    quantize_static(str(prep_path), str(int8_path), Calib(data["val"]["X"][:300]),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    torch.set_num_threads(1)
    with torch.no_grad():
        torch_lat = latency_ms(lambda: model(x1))
    res = dict(params=n_params(model), torch_latency_ms=torch_lat)
    X_te = data["test"]["X"][:, None]
    torch_logits = predict_logits(model, data["test"]["X"]).numpy()
    for name, path in [("onnx_fp32", fp32_path), ("onnx_int8", int8_path)]:
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        logits = np.concatenate([sess.run(None, {"x": X_te[i:i + 200]})[0] for i in range(0, len(X_te), 200)])
        res[name] = dict(
            size_kb=path.stat().st_size / 1024,
            latency_ms=latency_ms(lambda: sess.run(None, {"x": X_te[:1]})),
            acc=float((logits.argmax(1) == data["test"]["y"]).mean()),
            max_abs_logit_diff=float(np.abs(logits - torch_logits).max()),
            agreement=float((logits.argmax(1) == torch_logits.argmax(1)).mean()),
        )
    torch.set_num_threads(torch.get_num_interop_threads())
    return res


def plot_all(data, ens, base, seed_metrics, ens_metrics, base_metrics, rel):
    names = [c.replace("_", "\n") for c in CLASSES]

    # Confusion matrix (ensemble)
    cm = confusion_matrix(data["test"]["y"], ens["test"].argmax(1), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="w" if cm[i, j] > 0.5 else "k", fontsize=9)
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True", title="Confusion matrix (CNN 3-seed ensemble, test)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion.png", dpi=120)

    # Accuracy vs SNR
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    seed_curves = np.array([m["snr_acc"] for m in seed_metrics])
    ax.fill_between(SNR_SWEEP_DB, seed_curves.min(0), seed_curves.max(0), color="C0", alpha=0.2, label="CNN seeds (min-max)")
    ax.plot(SNR_SWEEP_DB, ens_metrics["snr_acc"], "o-", color="C0", label="CNN ensemble")
    ax.plot(SNR_SWEEP_DB, base_metrics["snr_acc"], "s--", color="C1", label="Features + RF")
    ax.axvspan(-5, 20, color="0.92", zorder=0, label="training SNR range")
    ax.axhline(1 / len(CLASSES), ls=":", color="0.5")
    ax.set(xlabel="Per-sample SNR [dB]", ylabel="Accuracy", ylim=(0, 1.02), title="Accuracy vs SNR")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "snr_curve.png", dpi=120)

    # Reliability diagram (seed 0, before / after temperature scaling)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    y = data["test"]["y"]
    for ax, (title, probs) in zip(axes, rel.items()):
        conf, correct = probs.max(1), probs.argmax(1) == y
        edges = np.linspace(0, 1, 16)
        mids, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                mids.append(conf[m].mean())
                accs.append(correct[m].mean())
        ax.plot([0, 1], [0, 1], ":", color="0.5")
        ax.plot(mids, accs, "o-")
        ax.set(xlabel="Confidence", title=f"{title}  (ECE {ece(probs, y):.3f})")
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(RESULTS / "reliability.png", dpi=120)


def main():
    data = {s: load(s) for s in ["val", "test", "snr_sweep", "ood"]}
    seeds = sorted(int(p.stem.split("seed")[1]) for p in CKPT.glob("cnn_seed*.pt"))
    models, seed_metrics, seed_probs, temps = {}, [], [], []
    for seed in seeds:
        model = MicroDopplerCNN(len(CLASSES))
        model.load_state_dict(torch.load(CKPT / f"cnn_seed{seed}.pt"))
        models[seed] = model.eval()
        logits = {s: predict_logits(model, d["X"]).numpy() for s, d in data.items()}
        T = fit_temperature(logits["val"], data["val"]["y"])
        probs = {s: softmax(z / T) for s, z in logits.items()}
        m = summarize_probs(probs, data, logits)
        m["ece_raw"] = float(ece(softmax(logits["test"]), data["test"]["y"]))
        m["temperature"] = float(T)
        seed_metrics.append(m)
        seed_probs.append(probs)
        temps.append(T)
        if seed == seeds[0]:
            rel = {"Raw softmax": softmax(logits["test"]), f"Temperature scaled (T={T:.2f})": probs["test"]}
        print(f"seed {seed}: acc {m['acc']:.3f}  ctrl {m['bal_acc_ctrl']:.3f}  ECE {m['ece_raw']:.3f}->{m['ece']:.3f}  "
              f"AUROC {m['auroc_msp']:.3f}")

    ens = {s: np.mean([p[s] for p in seed_probs], 0) for s in data}
    ens_metrics = summarize_probs(ens, data)
    base = dict(np.load(RESULTS / "baseline_probs.npz"))
    base_metrics = summarize_probs(base, data)
    deploy = deployment(models[seeds[0]], data)
    plot_all(data, ens, base, seed_metrics, ens_metrics, base_metrics, rel)

    # Per-class recall and unknown-target predictions
    recall = lambda p: confusion_matrix(data["test"]["y"], p["test"].argmax(1), normalize="true").diagonal()
    ood_pred = np.bincount(ens["ood"].argmax(1), minlength=len(CLASSES)) / len(ens["ood"])

    def ms(key, fmt="{:.3f}"):
        v = np.array([m[key] for m in seed_metrics])
        return f"{fmt.format(v.mean())} ± {fmt.format(v.std())}"

    lines = [f"# Evaluation (seeds {seeds}; test n={len(data['test']['y'])}, OOD n={len(data['ood']['y'])})\n",
             "## Main metrics\n",
             f"| Model | Test acc | Macro F1 | Balanced acc, \\|v\\|<={V_CONTROLLED:g} m/s | ECE (raw → TS) | Unknown AUROC (MSP) | FPR@95TPR (MSP) |",
             "|---|---|---|---|---|---|---|",
             f"| CNN (mean ± std over seeds) | {ms('acc')} | {ms('f1')} | {ms('bal_acc_ctrl')} | {ms('ece_raw')} → {ms('ece')} | "
             f"{ms('auroc_msp')} | {ms('fpr95_msp')} |",
             f"| CNN 3-seed ensemble | {ens_metrics['acc']:.3f} | {ens_metrics['f1']:.3f} | {ens_metrics['bal_acc_ctrl']:.3f} | "
             f"{ens_metrics['ece']:.3f} | {ens_metrics['auroc_msp']:.3f} | {ens_metrics['fpr95_msp']:.3f} |",
             f"| Features + RF | {base_metrics['acc']:.3f} | {base_metrics['f1']:.3f} | {base_metrics['bal_acc_ctrl']:.3f} | "
             f"{base_metrics['ece']:.3f} | {base_metrics['auroc_msp']:.3f} | {base_metrics['fpr95_msp']:.3f} |",
             f"\nEnergy score (CNN): AUROC {ms('auroc_energy')}, FPR@95TPR {ms('fpr95_energy')}. "
             f"Temperatures: {', '.join(f'{t:.2f}' for t in temps)}\n",
             "## Per-class recall\n",
             "| Model | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES),
             "| CNN ensemble | " + " | ".join(f"{r:.3f}" for r in recall(ens)) + " |",
             "| Features + RF | " + " | ".join(f"{r:.3f}" for r in recall(base)) + " |",
             f"\n## Unknown target ({OOD_CLASS}) — what the ensemble calls it\n",
             "| " + " | ".join(CLASSES) + " |", "|" + "---|" * len(CLASSES),
             "| " + " | ".join(f"{p:.1%}" for p in ood_pred) + " |",
             "\n## Accuracy vs SNR\n",
             "| Model | " + " | ".join(f"{s} dB" for s in SNR_SWEEP_DB) + " |", "|---|" + "---|" * len(SNR_SWEEP_DB),
             "| CNN ensemble | " + " | ".join(f"{a:.3f}" for a in ens_metrics["snr_acc"]) + " |",
             "| Features + RF | " + " | ".join(f"{a:.3f}" for a in base_metrics["snr_acc"]) + " |",
             "\n## Deployment (seed-0 model, single CPU thread, batch 1)\n",
             f"Parameters: {deploy['params']:,}. PyTorch latency: median {deploy['torch_latency_ms'][0]:.2f} ms "
             f"(p99 {deploy['torch_latency_ms'][1]:.2f} ms)\n",
             "| Format | Size [KB] | Median latency [ms] | p99 [ms] | Test acc | Max \\|Δlogit\\| vs PyTorch | Top-1 agreement |",
             "|---|---|---|---|---|---|---|"]
    for name in ["onnx_fp32", "onnx_int8"]:
        d = deploy[name]
        lines.append(f"| {name} | {d['size_kb']:.0f} | {d['latency_ms'][0]:.2f} | {d['latency_ms'][1]:.2f} | {d['acc']:.3f} | "
                     f"{d['max_abs_logit_diff']:.4f} | {d['agreement']:.3f} |")
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / "metrics.json").write_text(json.dumps(dict(seeds=seed_metrics, ensemble=ens_metrics,
                                                          baseline=base_metrics, deployment=deploy,
                                                          ood_pred=ood_pred.tolist()), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
