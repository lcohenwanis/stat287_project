import os
import copy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.utils.class_weight import compute_class_weight

# !pip3 install seaborn
# import seabron as sns
from PIL import Image
import matplotlib.pyplot as plt


def load_datasets():
    # Get the image file directories
    train_dir = os.path.join('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/chest_xray/train')
    test_dir = os.path.join('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/chest_xray/test')
    validation_dir = os.path.join('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/chest_xray/val')

    train_names_normal = os.listdir(os.path.join(train_dir,'NORMAL'))
    train_names_pneumonia = os.listdir(os.path.join(train_dir,'PNEUMONIA'))

    test_names_normal = os.listdir(os.path.join(test_dir,'NORMAL'))
    test_names_pneumonia = os.listdir(os.path.join(test_dir,'PNEUMONIA'))

    val_names_normal = os.listdir(os.path.join(validation_dir,'NORMAL'))
    val_names_pneumonia = os.listdir(os.path.join(validation_dir,'PNEUMONIA'))

    # Create datasets for training, testing, and validation
    IMAGE_SIZE = (300, 300)

    train_data = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir, 
                                                                    image_size=IMAGE_SIZE,
                                                                    label_mode='binary', 
                                                                    batch_size=64,
                                                                    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                    image_size=IMAGE_SIZE,
                                                                    batch_size=64,
                                                                    label_mode="binary",shuffle=False)


    val_data = tf.keras.preprocessing.image_dataset_from_directory(directory=validation_dir,
                                                                    image_size=IMAGE_SIZE,
                                                                    batch_size=64,
                                                                    label_mode="binary", shuffle=False)
    
    return train_data, test_data, val_data


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))
    
    return s


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    # compute activation
    # compute cost by using np.dot to perform multiplication. 
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m)*np.sum((Y*np.log(A)) + (1-Y)*np.log(1-A))
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1/m)*(np.dot(X, (A-Y).T))
    db = (1/m)*np.sum(A-Y)
    
    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction




def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros 
    w = np.zeros((X_train.shape[0], 1))
    b = 0.0

    
    # Gradient descent 
    params, grads, costs = optimize(w, b, X_train, Y_train,
                                    num_iterations=num_iterations, learning_rate=learning_rate, print_cost=print_cost)
    
    # Retrieve parameters w and b from dictionary "params"
    w = params['w']
    b = params['b']
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


def main():
    # Load data from images
    train_data, test_data, val_data = load_datasets()

    # Extract labels from tf BatchDatasets
    Y_train = np.concatenate([y for x, y in train_data], axis=0)
    Y_test = np.concatenate([y for x, y in test_data], axis=0)

    # Extract data from tf BatchDatasets
    X_train = np.concatenate([x for x, y in train_data], axis=0)
    X_test = np.concatenate([x for x, y in test_data], axis=0)

    # Train  model
    log_reg_model = model(X_train, Y_train, X_test, Y_test,
                   num_iterations=2000, learning_rate=0.5, print_cost=False)
 
    # Plot learning curve (with costs)
    costs = np.squeeze(log_reg_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(log_reg_model["learning_rate"]))
    plt.savefig('/Users/luctheduke/Desktop/UVM Grad School/Year 2/STAT 287 - DS 1/stat287_project/images/log_reg_cost_per_it.png')

if __name__ == "__main__":
    main()

