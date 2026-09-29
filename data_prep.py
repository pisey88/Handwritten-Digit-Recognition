from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Download MNIST CSV dataset via kagglehub
DATA_DIR = Path(kagglehub.dataset_download("oddrationale/mnist-in-csv"))
CLASS_NAMES = [str(i) for i in range(10)]

def load_csv(filename):
    file_path = DATA_DIR / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Could not find dataset file: {file_path}")

    dataframe = pd.read_csv(file_path)
    if "label" not in dataframe.columns or dataframe.shape[1] != 785:
        raise ValueError(f"{filename} must contain a label column and 784 pixel columns.")

    pixel_columns = dataframe.drop(columns="label")
    X = pixel_columns.to_numpy(dtype=np.float32) / 255.0
    y = dataframe["label"].astype("int64")

    return X, y, pixel_columns.columns

X_train, y_train, train_columns = load_csv("mnist_train.csv")
X_test, y_test, test_columns = load_csv("mnist_test.csv")

if not train_columns.equals(test_columns):
    raise ValueError("Training and test pixel columns must have the same order.")

if __name__ == "__main__":
    print("Dataset Shapes:")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("Pixel min:", X_train.min(), "Pixel max:", X_train.max())
