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

Part 2: Model and TrainingWhy does the model use 784 inputs and 10 outputs?784 Inputs: Matches the flattened feature vector size ($28 \times 28 = 784$ pixels) per image vector.10 Outputs: Corresponds to the 10 distinct target classes (0–9). The linear layer computes 10 unnormalized raw confidence scores (logits) per image:$$\text{logits} = X \cdot W + b$$Where weight matrix $W$ has dimensions $(784, 10)$ and bias vector $b$ has dimensions $(10,)$.Why do we use CrossEntropyLoss?CrossEntropyLoss is the standard loss function for multiclass classification tasks in PyTorch because it:Combines LogSoftmax and NLLLoss (Negative Log Likelihood Loss) into a single, numerically stable calculation.Converts raw output logits into predicted class probability distributions.Heavily penalizes the model when it assigns a low probability to the correct ground-truth digit label.What batch size, learning rate, and number of epochs did you use?Batch Size: 64Learning Rate: 0.001 (using Adam optimizer with weight_decay = 0.0001)Epochs: 20Random Seed: 42How did the training loss change? What does this tell you?The training loss dropped steadily across all 20 epochs (starting around ~0.45 and declining smoothly down to ~0.27). This consistent downward trajectory confirms that gradient descent successfully minimized classification errors and optimized linear weights without oscillating or diverging.Model Architecture & Training SettingsPlaintextMNISTLogisticRegression(
  (linear): Linear(in_features=784, out_features=10, bias=True)
)
Total Parameters: 7,850 (784 * 10 weights + 10 biases)
Training Settings:Optimizer: Adam ($\text{lr} = 0.001$, $\text{weight\_decay} = 0.0001$)Loss Function: CrossEntropyLoss()Batch Size: 64Epochs: 20Explanation: The smooth downward trajectory of the loss over 20 epochs shows that backpropagation effectively tuned the linear decision boundaries between all 10 digit classes.
