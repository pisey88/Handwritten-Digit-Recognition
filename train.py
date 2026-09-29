from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

from data_prep import X_train, X_test, y_train, y_test, CLASS_NAMES, BASE_DIR

torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Save sample digits figure (0-9)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for digit in range(10):
    idx = np.where(y_train == digit)[0][0]
    ax = axes[digit // 5, digit % 5]
    ax.imshow(X_train[idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {digit}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_digits.png", dpi=150)
plt.close()

# Datasets and Loaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model Definition
class MNISTLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

model = MNISTLogisticRegression(784, 10).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)

# Training Loop
train_losses = []
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    avg_loss = running_loss / len(train_dataset)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] Loss: {avg_loss:.4f}")

# Save Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("MNIST: Multiclass Logistic Regression Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
plt.close()

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        preds = logits.argmax(dim=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

# Baseline Evaluation
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

(OUTPUT_DIR / "evaluation.txt").write_text(
    f"Test Accuracy: {accuracy:.4f}\nBaseline Accuracy: {baseline_acc:.4f}\n\n" + report
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Predictions Figure
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    color = "green" if y_pred[i] == y_true[i] else "red"
    ax.set_title(f"True: {y_true[i]} | Pred: {y_pred[i]}", color=color)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions.png", dpi=150)
plt.close()

# Misclassified Figure
misclassified_idx = np.where(y_pred != y_true)[0][:3]
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for i, idx in enumerate(misclassified_idx):
    axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
    axes[i].set_title(f"True: {y_true[idx]} | Pred: {y_pred[idx]}", color="red")
    axes[i].axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "misclassified.png", dpi=150)
plt.close()

# Save Model
torch.save(model.state_dict(), OUTPUT_DIR / "mnist_logistic_model.pth")
print("\nTraining complete! All artifacts saved to outputs/")
