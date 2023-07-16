# Predicting The Next Word - NLP

Welcome to my Predicting The Next Word project! This project focuses on generating Shakespearean sonnets using a language model based on LSTM (Long Short-Term Memory) neural networks. The model is trained on a dataset of Shakespearean sonnets and can generate new text based on a given seed.

## Overview
The project utilizes the TensorFlow and Keras libraries to build and train the LSTM-based language model. It begins by preprocessing the Shakespearean sonnets dataset, including tokenizing the text and generating input sequences. The input sequences are then padded to ensure uniform length.

Next, a sequential model is created with an embedding layer, bidirectional LSTM layer, and a dense layer with softmax activation. The model is trained on the input sequences and corresponding labels (one-hot encoded). The training process involves minimizing the categorical cross-entropy loss using the Adam optimizer.

Once the model is trained, it can be used to generate new text. By providing a seed text, the model predicts the next word and appends it to the seed. This process is repeated for a desired number of words, resulting in the generation of Shakespearean-style sonnets.

The project includes visualizations of the training accuracy and loss curves, allowing for an analysis of the model's performance during training.

Overall, this project provides a practical example of using LSTM neural networks for text generation, specifically focusing on the generation of Shakespearean sonnets.

## Dependencies
To run this project, you need to have the following dependencies installed:

- numpy
- matplotlib
- tensorflow

## Dataset
Dataset can be downloaded through this link: https://www.opensourceshakespeare.org/views/sonnets/sonnet_view.php?range=viewrange&sonnetrange1=1&sonnetrange2=154

## Contact
For any inquiries or questions, please contact bu@zainrhamid.com.
Happy coding!
