import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms



class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        gates        = torch.sigmoid(self.gate_scores)
        gated_weight = self.weight * gates.unsqueeze(1)
        return F.linear(x, gated_weight, self.bias)


class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PrunableLinear(3072, 512)
        self.layer2 = PrunableLinear(512,  256)
        self.layer3 = PrunableLinear(256,  10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def get_gates(self):
        return {
            'layer1': torch.sigmoid(self.layer1.gate_scores).detach(),
            'layer2': torch.sigmoid(self.layer2.gate_scores).detach(),
            'layer3': torch.sigmoid(self.layer3.gate_scores).detach(),
        }

    def sparsity_loss(self):
        all_gates = torch.cat([
            torch.sigmoid(self.layer1.gate_scores),
            torch.sigmoid(self.layer2.gate_scores),
            torch.sigmoid(self.layer3.gate_scores),
        ])
        return all_gates.mean()



def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



def compute_sparsity(gate_values, threshold=1e-2):
    if isinstance(gate_values, torch.Tensor):
        gate_values = gate_values.detach().cpu().numpy()
    gate_values = np.asarray(gate_values).flatten()
    sparsity = (gate_values < threshold).sum() / len(gate_values) * 100
    return round(float(sparsity), 2)


def compute_metrics(gate_values, accuracy):
    if isinstance(gate_values, torch.Tensor):
        gate_values = gate_values.detach().cpu().numpy()
    gate_values = np.asarray(gate_values).flatten()
    return {
        "accuracy":     round(float(accuracy) * 100, 2),   
        "sparsity_pct": compute_sparsity(gate_values),
        "mean_gate":    round(float(gate_values.mean()), 4),
        "std_gate":     round(float(gate_values.std()),  4),
    }



def extract_all_gates(model):
    gate_arrays = []
    for module in model.modules():
        if hasattr(module, 'gate_scores'):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            gate_arrays.append(gates.flatten())
    return np.concatenate(gate_arrays) if gate_arrays else np.array([])



def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            _, predicted   = outputs.max(1)
            correct       += predicted.eq(labels).sum().item()
            total         += labels.size(0)
    return 100.0 * correct / total


def evaluate_with_gates(model, loader, device):
    acc_pct     = evaluate(model, loader, device)
    gate_values = extract_all_gates(model)
    return acc_pct / 100.0, gate_values



def train(epochs=10, lambda_sparse=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    train_loader, test_loader = get_cifar10_loaders()

    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs   = model(images)
            cls_loss  = criterion(outputs, labels)
            spar_loss = model.sparsity_loss()
            loss      = cls_loss + lambda_sparse * spar_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc      = evaluate(model, test_loader, device)

        gates      = model.get_gates()
        mean_gates = {k: v.mean().item() for k, v in gates.items()}

        print(f"  Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%  "
              f"| Gates → L1: {mean_gates['layer1']:.4f}  "
              f"L2: {mean_gates['layer2']:.4f}  "
              f"L3: {mean_gates['layer3']:.4f}")

    return model, test_loader, device



def build_results_table(lambda_values, epochs=10):
    rows    = []
    records = []

    for lam in lambda_values:
        print(f"\n── Training λ = {lam} {'─'*40}")
        model, test_loader, device = train(epochs=epochs, lambda_sparse=lam)

        acc, gate_values = evaluate_with_gates(model, test_loader, device)
        m = compute_metrics(gate_values, acc)

        print(f"  → Accuracy: {m['accuracy']:.2f}%  |  Sparsity: {m['sparsity_pct']:.2f}%")

        rows.append({
            "Lambda":     lam,
            "Accuracy":   f"{m['accuracy']:.2f} %",
            "Sparsity %": f"{m['sparsity_pct']:.2f} %",
        })
        records.append({
            "lambda":      lam,
            "accuracy":    m["accuracy"],
            "sparsity":    m["sparsity_pct"],
            "gate_values": gate_values,
        })

    df = pd.DataFrame(rows)
    print("\n── Results Table ──────────────────────────────")
    print(df.to_string(index=False))
    return df, records



def select_best(records):
    sorted_recs = sorted(records, key=lambda r: (-r["accuracy"], -r["sparsity"]))
    best = sorted_recs[0]
    print(f"\n── Best model: λ = {best['lambda']}  "
          f"(Acc: {best['accuracy']:.2f}%  Sparsity: {best['sparsity']:.2f}%)")
    return best



def plot_gate_histogram(gate_values, lambda_val=None, save_path=None):
    gate_values = np.asarray(gate_values).flatten()

    fig, ax = plt.subplots(figsize=(8, 4.5))

    threshold = 1e-2
    pruned = gate_values[gate_values <  threshold]
    active = gate_values[gate_values >= threshold]

    ax.hist(pruned, bins=40, range=(0, 1), color="#E8593C",
            alpha=0.85, label=f"Pruned  (< {threshold})", edgecolor="white",
            linewidth=0.4)
    ax.hist(active, bins=40, range=(0, 1), color="#3B8BD4",
            alpha=0.75, label="Active  (≥ 1e-2)", edgecolor="white",
            linewidth=0.4)

    ax.axvline(threshold, color="#888", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold}")

    lam_str = f"  (λ = {lambda_val})" if lambda_val is not None else ""
    ax.set_title(f"Distribution of Sigmoid Gate Values{lam_str}",
                 fontsize=13, fontweight="medium", pad=12)
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count",      fontsize=11)
    ax.legend(framealpha=0.9,   fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    sp = compute_sparsity(gate_values)
    ax.text(0.97, 0.95, f"Sparsity: {sp:.1f} %",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.9))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram → {save_path}")
    plt.show()
    return fig


def plot_lambda_comparison(records, save_path=None):
    lambda_values = [r["lambda"]   for r in records]
    accuracies    = [r["accuracy"] for r in records]
    sparsities    = [r["sparsity"] for r in records]

    x      = np.arange(len(lambda_values))
    width  = 0.38
    labels = [str(l) for l in lambda_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    bars1 = ax1.bar(x, accuracies, width=width, color="#3B8BD4",
                    edgecolor="white", linewidth=0.5)
    ax1.set_title("Test Accuracy vs Lambda", fontsize=12, fontweight="medium")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_xlabel("Lambda (L1 strength)"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    bars2 = ax2.bar(x, sparsities, width=width, color="#E8593C",
                    edgecolor="white", linewidth=0.5)
    ax2.set_title("Sparsity % vs Lambda", fontsize=12, fontweight="medium")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_xlabel("Lambda (L1 strength)"); ax2.set_ylabel("Sparsity (%)")
    ax2.set_ylim(0, 105)
    ax2.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    plt.suptitle("Effect of L1 Regularization Strength on Pruning",
                 fontsize=13, fontweight="medium", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison plot → {save_path}")
    plt.show()
    return fig



if __name__ == "__main__":
    LAMBDA_VALUES = [0.0001, 0.001, 0.01]
    EPOCHS        = 10          

    df, records = build_results_table(LAMBDA_VALUES, epochs=EPOCHS)

    best = select_best(records)

    plot_gate_histogram(
        best["gate_values"],
        lambda_val=best["lambda"],
        save_path="gate_histogram.png",
    )

    plot_lambda_comparison(records, save_path="lambda_comparison.png")

    print("\nDone.")
