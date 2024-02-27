# deep_learning_challenge

Overview

This challenge was undertaken to develop a tool that the nonprofit foundation Alphabet Soup could use to assist with selecting applicants for funding who would have the highest likelihood of success in their ventures.

Data for this challenge covered more than 34,000 organizations that have received funding in years past from Alphabet Soup. The data came in the form of a CSV file from Alphabet's Business team and was structured with several columns that captured metadata about each organization.

The challenge was completed in three steps, using two different notebooks in Google Colab.

In the first step, the data was read into a notebook and pre-processed by dropping columns, conducting value counts of variables in individual columns, and binning those variables. The operator pd.get_dummies() was used to encode categorical variables, followed by splitting the data into a features array (X) and a target array (y). The train_test_split function was then employed to split the data into training and testing sets, and those sets scaled with the scaler instance fitted to the training data.

In the second step, a neural network was designed using TensorFlow, making a binary classification model to predict if an Alphabet Soup-funded organization would be successful based on its features in the dataset. Hidden and output layers were made with activation functions, and the model then compiled and trained. The model was then evaluated using test data to determine loss and accuracy. Results were finally saved and exported to an HDF5 file.

In the third step, attempts were made to optimize the model developed in the second step, including re-processing of data and making changes in layers and features in training the model. The goal of optimization was to reach a predictive accuracy higher than 75%.

Results

Results were as follows.

Data Preprocessing

Target variables - the "IS_SUCCESSFUL" column
Feature variables - all other surviving columns than the target column after pre-processing
Variables removed - "EIN", "NAME", and "ASK_AMT" were removed (latter in optimization phase)

Compiling, Training, and Evaluating the Model

Original Model:
First layer - 43 input dimensions, 80 units, activation = 'relu'
Second layer - 30 units, activation = 'relu'
Output layer - 1 unit, activation = 'sigmoid'
Epochs = 100
Loss - 0.5784
Accuracy - 0.7298

Optimization Attempt 1:
First layer - 24 input dimensions, 80 units, activation = 'relu'
Second layer - 30 units, activation = 'relu'
Output layer - 1 unit, activation = 'sigmoid'
Epochs = 100
Loss - 0.5887
Accuracy - 0.7127

Optimization Attempt 2:
First layer - 22 input dimensions, 80 units, activation = 'relu'
Second layer - 40 units, activation = 'relu'
Third layer - 20 units, activation = 'relu'
Output layer - 1 unit, activation = 'sigmoid'
Epochs = 200
Loss - 0.5887
Accuracy - 0.7118

Optimization Attempt 3:
First layer - 22 input dimensions, 70 units, activation = 'relu'
Second layer - 50 units, activation = 'relu'
Third layer - 30 units, activation = 'relu'
Output layer - 1 unit, activation = 'sigmoid'
Epochs = 300
Loss - 0.5960
Accuracy - 0.7095

Summary

Although more refinement of the data was accomplished with continued pre-processing, achieving a reduction in the number of dimensions, neither that nor adding another hidden layer nor still increasing the number of epochs had a corresponding effect on accuracy. Despite the measures taken to optimize the model, the changes implemented actually resulted in an increase in loss and an erosion in accuracy.

This experience highlights the difficulty with determining the optimal layer components and number of epochs to use. Although the dataset was thoroughly processed for more efficient performance, and the number of layers increased to three (the level that many data scientists believe sufficient for most complex interactions), results with each attempt at optimization not only did not approach the target of 75% accuracy but actually decreased with each subsequent attempt.

Alteration of the activation functions used may be the key in this case to achieve the optimization target. And to determine a better combination of those functions, use of the KerasTuner could be employed to select the activation function for each hidden layer. The KerasTuner could also decide the number of neurons for the first dense layer as well as the number itself of layers. The tuner search could then produce new models with which to compare predictive accuracy.
