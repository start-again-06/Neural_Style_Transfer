# 🎨 Neural Style Transfer with VGG19

This project implements **Neural Style Transfer** using a pre-trained VGG19 network. The goal is to create a new image that preserves the content of one image and the style of another.

---

## 📌 Overview

- **Framework**: TensorFlow 2.x & Keras
- **Model**: Pre-trained VGG19 (feature extraction only)
- **Objective**: Generate an image that matches content from a content image and style from a style image

---

## 🧠 Key Concepts

### 🖼️ Content Cost
Captures how different the content of the generated image is from the content image.

### 🎨 Style Cost
Captures how different the style (textures/colors) of the generated image is from the style image, using **Gram Matrices**.

### 🔀 Total Cost
Weighted sum of content and style costs:
```python
J = alpha * J_content + beta * J_style
```

---

## 🏗️ Architecture
- Uses VGG19 without the fully connected top layers
- Only specific convolution layers are used for computing style/content representations

### 🔹 Style Layers
```python
STYLE_LAYERS = [
    ('block1_conv1', 1.0),
    ('block2_conv1', 0.8),
    ('block3_conv1', 0.7),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.1)
]
```

### 🔸 Content Layer
```python
CONTENT_LAYER = [('block5_conv4', 1)]
```

---

## ⚙️ Usage Steps

1. **Load and Preprocess Images**
2. **Extract Features Using VGG19**
3. **Compute Content and Style Costs**
4. **Define Total Cost Function**
5. **Optimize Generated Image**

---

## 🔄 Training Process

- Initialize generated image (content + noise)
- Compute gradients of total loss with respect to the generated image
- Apply optimization (Adam)
- Save outputs every 250 epochs

---

## 📚 References

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge – [A Neural Algorithm of Artistic Style (2015)](https://arxiv.org/abs/1508.06576)
- Harish Narayanan – [Convolutional Neural Networks for Artistic Style Transfer](https://harishnarayanan.org/writing/artistic-style-transfer/)
- Log0 – [TensorFlow Implementation of Neural Style Transfer](https://github.com/log0/neural-style-tf)
- Karen Simonyan, Andrew Zisserman – [Very Deep Convolutional Networks for Large-Scale Image Recognition (2015)](https://arxiv.org/abs/1409.1556)
- [MatConvNet – CNNs for MATLAB](http://www.vlfeat.org/matconvnet/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications: VGG19](https://keras.io/api/applications/vgg/)

---
