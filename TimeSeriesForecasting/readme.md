# Time-Series Forecasts with Convolution Over LSTM

I'm excited to share my latest GitHub project focusing on time series forecasting using the power of TensorFlow and an innovative approach involving Convolutional over Long Short-Term Memory (LSTM) layers. In this project, I delve into the intriguing realm of predicting daily minimum temperatures in Melbourne from 1981 to 1990, utilizing a combination of recurrent and convolutional neural network architectures.

## Overview

The heart of this project lies in the "Daily Minimum Temperatures in Melbourne" dataset, which contains a treasure trove of historical data capturing the daily minimum temperatures spanning a decade. Armed with TensorFlow's cutting-edge capabilities, I aim to harness the intrinsic patterns and correlations present within the data to accurately predict future temperature trends.

## Part 1: Preprocessing Data

Initially, code reads a CSV file named ‘daily-min-temperatures.csv’ using the Pandas library and stores the data in two lists, time and temperatures. The code then converts these lists into NumPy arrays named TIME and SERIES. Code also set some constants such as SPLIT_TIME, WINDOW_SIZE, BATCH_SIZE, and SHUFFLE_BUFFER_SIZE.

## Part 2: Some Basics Functions

In the next step, code contains some functions which are used later in the code. Functions are as follow, 
 - **plot_series:** The function first plots the time series using the plt.plot() function. It then sets the labels for the x-axis and y-axis, and adds a grid to the plot.
 - **train_val_split:** The function first splits the time series into two parts, the training set and the validation set. The training set is the first time_step time steps, and the validation set is the remaining time steps.
 - **windowed_dataset:** The function first creates a dataset of windows from the time series. The window size is the number of time steps in each window. The function then shuffles the data and batches it into groups of batch_size windows. The function then prefetches the next batch of data so that it is ready to be processed without having to wait for it to be loaded from disk.
 - **compute_metrics:** The function first computes the MSE and MAE using the tf.keras.metrics.mean_squared_error() and tf.keras.metrics.mean_absolute_error() functions. It then returns the MSE and MAE as numpy arrays.

## Part 3: Model Architecture

The model consists of the following layers:

 - A convolutional layer with 64 filters, kernel size 3, and causal padding. The causal padding ensures that the output of the layer only depends on the past values of the input sequence.
 - Two LSTM layers with 64 units each. LSTM layers are a type of recurrent neural network that are well-suited for modeling time series data.
 - A dense layer with 30 units, followed by a ReLU activation function.
 - A dense layer with 10 units, followed by a ReLU activation function.
 - A dense layer with 1 unit, followed by a Lambda layer that multiplies the output by 400.
The model is trained using the Adam optimizer and the mean squared error loss function.

## Prerequisites

Before running the project, ensure that the following dependencies are installed:
 
- TensorFlow 
- NumPy 
- Matplotlib
- Pandas

## Dataset

Dataset can be found in the same directory

## Contact

For any inquiries or questions, please contact z@zainrhamid.com.

Stay curious!
