# Binary Neural Network: Handwritten Digit Classifier (0s vs 1s)

A Python implementation of a neural network built from scratch to classify handwritten digits 0 and 1. Uses synthetic data generation instead of MNIST downloads, implements forward/backward propagation with NumPy, and includes visualization tools for training progress and custom image testing.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Testing Custom Images](#testing-custom-images)
- [Code Documentation](#code-documentation)
- [Educational Value](#educational-value)
- [Requirements](#requirements)
- [License](#license)

## Overview

This project demonstrates the fundamental concepts of neural networks by implementing a binary classifier from scratch. Instead of relying on machine learning frameworks, it uses only NumPy for mathematical operations and Matplotlib for visualization, making it ideal for understanding the underlying mechanics of neural networks.

The network learns to distinguish between handwritten 0s and 1s using synthetic training data, achieving over 96% accuracy on test data.

## Features

- **Pure Python Implementation**: No ML frameworks - just NumPy and Matplotlib
- **Synthetic Data Generation**: Creates 2000 realistic digit images with natural variations
- **Complete Neural Network**: Forward propagation, backpropagation, and gradient descent
- **Real-time Visualization**: Training progress, loss curves, and accuracy metrics
- **Custom Image Testing**: Load and test your own handwritten digits
- **Comprehensive Documentation**: 100+ inline code annotations
- **Educational Focus**: Clear explanations of every concept and implementation detail

## Installation

### Prerequisites
```bash
# Required packages
pip install numpy matplotlib jupyter

# Optional (for custom image testing)
pip install Pillow
```

### Setup
1. Clone or download this repository
2. Open terminal in the project directory
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `binary_classifier.ipynb`

## Usage

### Basic Training
Run the notebook cells in order:

1. **Cells 1-3**: Generate synthetic data and visualize samples
2. **Cell 4**: Preprocess data (normalize and flatten images)
3. **Cell 5**: Create the neural network architecture
4. **Cells 6-7**: Train the network (takes ~30 seconds)
5. **Cell 8**: View training results and sample predictions

### Testing Custom Images
After training, use cells 9-11 to test your own images:

```python
# Load images from a folder
folder_path = "/path/to/your/images"
custom_imgs, img_names = load_custom_images_from_folder(folder_path)
test_custom_images(nn, custom_imgs, img_names)
```

## Project Structure

```
├── Cell 1: Data Generation (synthetic 0s and 1s)
├── Cell 2: Data Verification 
├── Cell 3: Sample Visualization
├── Cell 4: Data Preprocessing
├── Cell 5: Neural Network Class
│   ├── Forward propagation
│   ├── Backpropagation  
│   ├── Loss calculation
│   └── Prediction methods
├── Cell 6: Training Function
├── Cell 7: Training Execution
├── Cell 8: Results Visualization
├── Cell 9: Individual Image Testing
├── Cell 10: Custom Image Setup
└── Cell 11: Verification Testing
```

## How It Works

### Network Architecture
- **Input Layer**: 784 neurons (28×28 flattened images)
- **Hidden Layer**: 64 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation

### Key Components
- **Data Generation**: Creates synthetic 0s (circles) and 1s (vertical lines)
- **Preprocessing**: Normalizes pixel values (0-1) and flattens images
- **Forward Pass**: Computes predictions through the network
- **Loss Function**: Binary cross-entropy for classification
- **Backpropagation**: Calculates gradients using chain rule
- **Optimization**: Gradient descent with learning rate 0.1

### Training Process
The network trains for 100 epochs, seeing all 1600 training examples in each epoch. Progress is tracked through:
- Training loss (decreases over time)
- Training accuracy (increases over time)  
- Test accuracy (measures generalization)

## Results

**Typical Performance:**
- Training Accuracy: ~98%
- Test Accuracy: ~96%
- Training Time: ~30 seconds
- Network Parameters: 50,368 trainable weights

**Training Progression:**
- Epoch 0: ~50% accuracy (random guessing)
- Epoch 20: ~85% accuracy (pattern recognition begins)
- Epoch 60: ~95% accuracy (fine-tuning)
- Epoch 100: ~96% accuracy (converged)

## Testing Custom Images

### Supported Formats
- PNG, JPG, JPEG, BMP, TIFF
- Any size (automatically resized to 28×28)
- Color or grayscale (converted to grayscale)
- Any background (auto-inverted if needed)

### Best Practices
- Use high contrast (dark digit, light background)
- Center the digit in the image
- Make digits thick enough to survive 28×28 resize
- Use simple, clear handwriting style

### Example Usage
```python
# Test a single image
predicted_class, confidence = load_single_image('/path/to/digit.png')

# Test multiple images from folder
custom_imgs, names = load_custom_images_from_folder('/path/to/folder')
test_custom_images(nn, custom_imgs, names)
```

## Code Documentation

The code includes 100 numbered footnotes [1-100] that explain:
- Mathematical operations and their purposes
- Neural network concepts and implementations
- Data preprocessing steps
- Visualization techniques
- Error handling and edge cases

These annotations are referenced in the comprehensive documentation guide included with the project.

## Educational Value

This project is designed for learning:

**Concepts Demonstrated:**
- Supervised learning with labeled data
- Forward and backward propagation
- Gradient descent optimization  
- Activation functions (ReLU, Sigmoid)
- Loss functions (Binary cross-entropy)
- Overfitting detection and prevention
- Data preprocessing and normalization

**Skills Developed:**
- Implementing neural networks from scratch
- Understanding mathematical foundations
- Data visualization and analysis
- Image processing and handling
- Python programming best practices

## Requirements

**Minimum Requirements:**
- Python 3.8+
- NumPy 1.19+
- Matplotlib 3.3+
- Jupyter Notebook

**Optional:**
- Pillow (PIL) for custom image testing
- 4GB RAM (for comfortable training)
- Modern CPU (training takes 30 seconds on typical hardware)

**Tested On:**
- macOS, Windows, Linux
- Python 3.8, 3.9, 3.10, 3.11
- Jupyter Notebook, JupyterLab, VS Code

## License

MIT License - See LICENSE file for details.

---

**Educational Use:** This project is specifically designed for learning and teaching neural network fundamentals. Feel free to modify, extend, and use for educational purposes.
