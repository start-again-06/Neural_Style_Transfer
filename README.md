Neural Style Transfer with VGG19 (TensorFlow/Keras)  
A comprehensive computer vision project that implements Neural Style Transfer using a pre-trained VGG19 network in TensorFlow and Keras. The project focuses on generating a new image that preserves the semantic content of one image while adopting the artistic style of another through optimization-based feature matching.

## Features
- End-to-end neural style transfer pipeline  
- Content and style representation using deep CNN features  
- Style extraction using Gram matrix correlations  
- Weighted combination of content and style objectives  
- Image optimization using gradient-based methods  
- Modular and educational implementation  

## Model & Framework
- Model: VGG19 (pre-trained, feature extraction only)  
- Framework: TensorFlow 2.x / Keras  
- Task: Neural Style Transfer  
- Input: Content image and style image  
- Output: Stylized image combining content and style  

## Core Components
- Content cost computation from deep convolutional features  
- Style cost computation using Gram matrices  
- Total cost function combining content and style losses  
- Optimization loop to iteratively update the generated image  

## Loss Formulation

Total Loss:  
J = α · J_content + β · J_style


- Content loss measures similarity between generated and content image features  
- Style loss measures similarity between generated and style image textures  

## Architecture
- VGG19 network without fully connected layers  
- Selected convolutional layers for content and style representation  

## Style Layers
- block1_conv1  
- block2_conv1  
- block3_conv1  
- block4_conv1  
- block5_conv1  

## Content Layer
- block5_conv4  

## Training & Optimization
- Generated image initialized as content image plus noise  
- Gradients computed with respect to generated image  
- Optimization performed using Adam optimizer  
- Intermediate stylized images saved at regular intervals  

## Dependencies
- Python 3.x  
- TensorFlow 2.x  
- Keras  
- NumPy  
- Matplotlib  
- Pillow  

## References
- A Neural Algorithm of Artistic Style – Gatys et al.  
- Very Deep Convolutional Networks for Large-Scale Image Recognition – Simonyan & Zisserman  
- TensorFlow Documentation  
- Keras Applications: VGG19  

## License
This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.
