from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data_prep import X_test, y_test, BASE_DIR

class MNISTLogisticRegression(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

def predict_samples(num_samples=5):
    model_path = BASE_DIR / "outputs" / "mnist_logistic_model.pth"
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    model = MNISTLogisticRegression()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    samples = torch.tensor(X_test[indices], dtype=torch.float32)

    with torch.no_grad():
        logits = model(samples)
        predictions = logits.argmax(dim=1).numpy()

    print("\n--- Model Predictions ---")
    for i, idx in enumerate(indices):
        print(f"Sample {i+1} | True Label: {y_test.iloc[idx]} | Model Prediction: {predictions[i]}")

if __name__ == "__main__":
    predict_samples()
