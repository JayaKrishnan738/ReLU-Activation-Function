# ReLU-Activation-Function


# README.md
"""
# Activation Functions Implementation

This repository contains the implementation of popular activation functions used in machine learning and neural networks. Included functions are:
- Sigmoid
- Stable Sigmoid
- ReLU (Rectified Linear Unit)

## The Role of Activation Functions

### Sigmoid Function

#### Probabilistic Interpretation
The sigmoid function outputs values in the range (0, 1), which can be interpreted as probabilities. For example, given an input \( x \), the output of the sigmoid function can represent the likelihood of a binary event occurring. This makes it particularly useful in models where the output needs to represent probabilities, such as classification tasks.

#### Logistic Regression
In logistic regression, the sigmoid function maps linear combinations of input features into a probability. The model predicts the probability of the dependent variable belonging to a certain class, typically class 1 in binary classification problems. The decision boundary is determined by a threshold (e.g., 0.5), where outputs greater than or equal to the threshold are classified as class 1, and outputs below are classified as class 0.

### ReLU (Rectified Linear Unit)

#### Explanation
ReLU is one of the most commonly used activation functions in deep learning. It outputs the input directly if it is positive; otherwise, it outputs zero. This property makes it simple yet effective for introducing non-linearity into the network.

#### Significance
1. **Efficient Computation:** ReLU is computationally efficient, as it involves simple thresholding.
2. **Sparsity:** ReLU sets negative values to zero, encouraging sparsity in the network's activations.
3. **Avoiding Vanishing Gradients:** Compared to sigmoid or tanh, ReLU avoids the vanishing gradient problem for positive inputs, enabling deeper networks to learn effectively.

Mathematically, ReLU is defined as:
\[ f(x) = \max(0, x) \]

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```

2. Navigate to the directory:
    ```bash
    cd activation_functions
    ```

3. Install the required libraries (if not already installed):
    ```bash
    pip install numpy
    ```

## Example Usage

### Running the Script

1. Run the script directly to input values:
    ```bash
    python activation_functions.py
    ```

2. Enter a single value or a list of comma-separated values when prompted. For example:
    ```
    Enter a number or a list of numbers (comma-separated): 1, -1, 0
    ```
    Output:
    ```
    Sigmoid result: [0.7310585786300049, 0.2689414213699951, 0.5]
    Stable Sigmoid result: [0.7310585786300049, 0.2689414213699951, 0.5]
    ReLU result: [1.0, 0.0, 0.0]
    ```

### Importing the Functions

```python
from activation_functions import sigmoid, sigmoid_stable, relu
```

### Compute ReLU for Single Values

```python
# Example: x = -3
result = relu(-3)
print("ReLU(-3):", result)
# Output: ReLU(-3): 0
```

### Compute ReLU for Arrays

```python
# Example: x = [1, -1, 0]
result = relu([1, -1, 0])
print("ReLU([1, -1, 0]):", result)
# Output: [1, 0, 0]
```

### Multi-Dimensional Arrays

```python
# Example: Multi-dimensional array
result = relu([[1, -2], [-3, 4]])
print("ReLU([[1, -2], [-3, 4]]):", result)
# Output: 
# [[1 0]
#  [0 4]]
```
