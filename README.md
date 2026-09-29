# Handwritten-Digit-Recognition

**Dataset and Preprocessing**

**How many training and test images are there?**

There are 60,000 training, 10,000 test.
- Training set: 60,000 images (mnist_train.csv).
- Test set: 10,000 images (mnist_test.csv).

**What are the image dimensions and class labels?**
Image Dimensions: 28x28 pixels in grayscale (1 color channel).
Class Labels: 10 discrete classes corresponding to numerical digits 0 through 9 ("0" to "9").

**Why do we convert each image into 784 values?**
A linear or logistic regression layer expects a 1D vector of numerical features per input sample. Flattening the 2D pixel matrix (28 pixels wide x 28 pixels high) transforms spatial pixel dimensions into a single array of 784 feature values (28 x 28 = 784).

**Why do we divide pixel values by 255?**
Raw pixel intensities range from `0` (black) to `255` (white). Dividing by `255.0` normalizes feature values to a floating-point range between `0.0` and `1.0`. Normalization:
1. Prevents feature scale disparities from causing numerical instability or exploding gradients during backpropagation.
2. Helps optimization algorithms (like Adam) converge faster and more reliably.


**Why must training and test data remain separate?**
The training set is used strictly to learn the model's parameters (weights $W$ and biases $b$). The test set acts as unseen data to evaluate how well the model **generalizes**. If test data leaks into training, evaluation metrics become artificially high, making it impossible to detect overfitting.


<img width="990" height="476" alt="image" src="https://github.com/user-attachments/assets/9cc505d3-8644-4aed-a14a-6073d80b097f" />

**Part 2: Model and Training**


