import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Subset
from jaxtyping import Float

class CNN(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=8, hidden_dim=16, num_classes=10, dropout=0.2):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3),
            torch.nn.Conv2d(out_channels, 2*out_channels, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2*out_channels*2*2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: Float[torch.Tensor, "batch channel height width"]):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = x.view(B, -1)
        x = self.linear(x)
        return x

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=512
)

base_out_channels = 8
base_hidden_dim = 16
base_lr = 1e-3

powers = 5
batch_size = 64
seeds = 3

for dropout in [0.0, 0.2]:
    computes = []
    accuracies = []
    losses = []
    model_sizes = []
    for model_size_pow in range(powers):
        for dataset_size_pow in range(powers):
            correct = 0
            total = 0
            validation_loss = 0
            for seed in range(seeds):
                random.seed(seed)
                torch.manual_seed(seed)
                model_out_channels = int(base_out_channels * (2 ** 0.5) ** model_size_pow)
                model_hidden_dim = int(base_hidden_dim * (2 ** 0.5) ** model_size_pow)

                dataset_size = int(len(train_dataset) / (2 ** dataset_size_pow))
                g = torch.Generator().manual_seed(seed * 10_000 + dataset_size_pow)
                idx = torch.randperm(len(train_dataset), generator=g)[:dataset_size].tolist()
                dataset = Subset(train_dataset, idx)

                train_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed * 10_000 + 123)
                )

                model = CNN(
                    in_channels=1,
                    out_channels=model_out_channels,
                    hidden_dim=model_hidden_dim,
                    num_classes=len(train_dataset.classes),
                    dropout=dropout
                )
                lr = base_lr * (base_out_channels / model_out_channels) ** 0.5
                optimizer = torch.optim.Adam(model.parameters(), lr)
                loss_function = torch.nn.CrossEntropyLoss()
                model_size = sum(p.numel() for p in model.parameters())

                model.train()
                for epoch in range(1):
                    for images, labels in train_loader:
                        optimizer.zero_grad()
                        logits = model(images)
                        loss = loss_function(logits, labels)
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    for images, labels in test_loader:
                        logits = model(images)
                        validation_loss += loss_function(logits, labels).item() * labels.shape[0]
                        preds = logits.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.shape[0]

            compute = dataset_size * model_size
            validation_accuracy = correct / total
            validation_loss /= total

            computes.append(compute)
            accuracies.append(validation_accuracy)
            losses.append(validation_loss)
            model_sizes.append(model_size)
            
            print(
                f"dropout={dropout} model_pow={model_size_pow} data_pow={dataset_size_pow} "
                f"out={model_out_channels} hid={model_hidden_dim} lr={lr:.2e} "
                f"params={model_size} dataset_size={dataset_size} acc={validation_accuracy:.4f}"
            )

    plt.figure()
    plt.scatter(computes, losses)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("compute (training examples * params)")
    plt.ylabel("validation loss")
    plt.savefig(f"loss_compute_{dropout}.png")

    plt.figure()
    plt.scatter(computes, accuracies)
    plt.xscale("log")
    plt.xlabel("compute (training examples * params)")
    plt.ylabel("accuracy")
    plt.savefig(f"accuracy_compute_{dropout}.png")

    zipped = list(zip(computes, losses, model_sizes))
    zipped.sort()
    min_loss = float("inf")
    frontier_compute = []
    frontier_model_size = []
    for compute, loss, model_size in zipped:
        if loss < min_loss:
            min_loss = loss
            frontier_compute.append(compute)
            frontier_model_size.append(model_size)

    plt.figure()
    plt.scatter(frontier_compute, frontier_model_size)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("compute (training examples * params)")
    plt.ylabel("model size")
    plt.savefig(f"model_size_compute_{dropout}.png")

    frontier_compute = np.array(frontier_compute)
    frontier_model_size = np.array(frontier_model_size)

    beta, intercept = np.polyfit(np.log(frontier_compute), np.log(frontier_model_size), 1)
    print(f"model_size_compute_exponent_{dropout}={beta}")
