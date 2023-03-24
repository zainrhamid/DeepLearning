# About

Welcome to my Neural Network's folder! Here, you will find my work on various deep learning projects using PyTorch and other neural network architectures.

## Projects

### Project 1: Image Segmentation with PyTorch

In this project, I worked with an image-mask dataset for segmentation. To handle the dataset, I created a custom dataset class using PyTorch that can load and preprocess the image and its corresponding mask.

I also used the albumentations library to apply segmentation augmentation to the images and masks. This helped to increase the diversity of the training data and improve the performance of the model.

To visualize the dataset, I plotted some image-mask pairs using Matplotlib.

For the segmentation model, I used a pre-trained state-of-the-art convolutional neural network (CNN) from the segmentation-models-pytorch library. Specifically, I used the UNet architecture, which is well-suited for image segmentation tasks.

Overall, this project helped me gain a better understanding of segmentation datasets and how to work with them using PyTorch and various libraries such as albumentations and segmentation-models-pytorch.
