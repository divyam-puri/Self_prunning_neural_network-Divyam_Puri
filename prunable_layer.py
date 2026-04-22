import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# ─── Layer ───────────────────────────────────────────────
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.ones(out_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        gated_weight = self.weight * gates.unsqueeze(1)
        return F.linear(x, gated_weight, self.bias)


# ─── Model ───────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = PrunableLinear(3072, 512)
        self.layer2 = PrunableLinear(512, 256)
        self.layer3 = PrunableLinear(256, 10)

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


# ─── Data ────────────────────────────────────────────────
def get_cifar10_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=transform)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                              download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ─── Evaluation ──────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ─── Training ────────────────────────────────────────────
def train(epochs=10, lambda_sparse=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    train_loader, test_loader = get_cifar10_loaders()

    model = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            cls_loss  = criterion(outputs, labels)         # classification loss
            spar_loss = model.sparsity_loss()              # sparsity penalty
            loss      = cls_loss + lambda_sparse * spar_loss  # combined

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc      = evaluate(model, test_loader, device)

        # Gate stats after each epoch
        gates = model.get_gates()
        mean_gates = {k: v.mean().item() for k, v in gates.items()}

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        print(f"          Gates → L1: {mean_gates['layer1']:.4f} | "
              f"L2: {mean_gates['layer2']:.4f} | L3: {mean_gates['layer3']:.4f}\n")

    print("Training done!")
    print("Final gate means:")
    for name, g in model.get_gates().items():
        active = (g > 0.5).sum().item()
        print(f"  {name}: {active}/{len(g)} neurons still active (gate > 0.5)")


if __name__ == "__main__":
    train(epochs=10, lambda_sparse=0.01)