import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    # It's like Wt X - y as the total error
    return np.square(X@w - y).mean()

###### Part 1.2 ######
def linear_regression_noreg(X, y):

    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here                    #
    #####################################################	
    return np.linalg.inv(X.T@X) @ X.T @ y
    

###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    return np.linalg.inv(X.T@X + lambd * np.identity(X.shape[1])) @ X.T @ y

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    ans = []
    for e in range(-14, 1):
        lambd = 2**e
        lambdi = lambd * np.identity(Xtrain.shape[1])
        #(XtX + Lambda I) * Xt * y
        w = np.linalg.inv(Xtrain.T@Xtrain + lambdi) @ Xtrain.T @ ytrain
        ans.append([mean_square_error(w, Xval, yval), lambd])
    return sorted(ans)[0][1]
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################	
    
    new_X = []
    for x in X:
      temp = []
      for exp in range(1, p+1): 
        for j in range(len(x)):
          temp.append(x[j]**exp)
      new_X.append(np.array(temp))         
        
    return np.array(new_X)

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

