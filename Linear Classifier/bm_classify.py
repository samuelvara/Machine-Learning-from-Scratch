import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = w0 if w0 else np.zeros(D)
    b = b0 if b0 else 0
    
    new_w = np.insert(w, 0, b, axis=0)
    new_X = np.insert(X, 0, 1, axis=1)
    # Simplify the problem into the usual Positive - Negative classifier
    new_y = np.where(y==0, -1, 1)
    
    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
		# derivative of the perceptron loss at 0)      # 
        ################################################
        for _ in range(max_iterations+1):
            pred_y = binary_predict(new_X, new_w, b=True)
            pred_y = np.where(pred_y==0, -1, 1)
            print(len(pred_y))
            #Error y * (Wt X)<--Pred y +1 if correctly classified else -1 the classic perceptron
            ywtx = new_y*pred_y
            ywtx = np.where(ywtx<=0, 1, 0)
            print(ywtx.shape, new_X.shape)
            new_w = new_w + (step_size/N) * ((ywtx*new_y)@new_X)

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    # 
        ################################################
        for _ in range(max_iterations+1):
            #Error sigmoid(-z) where z is our classic error yWtx
            ywtx = sigmoid(-new_y*(new_X@new_w))
            new_w = new_w + (step_size/N) * ((ywtx*new_y)@new_X) 
        

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    #cleaning W from bias at index 0, bias at index 0 of W
    return np.delete(new_w, 0), new_w[0]


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################
    
    return 1/(1+np.exp(-z))


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
        
    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    if b!=True:
        w = np.insert(w, 0, b, axis=0)
        X = np.insert(X, 0, 1, axis=1)
    predicted = np.sign(X@w)
    return np.where(predicted==-1, 0, 1)
    #assert preds.shape == (N,) 
    #return preds


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape
    
    new_X = np.append(X, np.ones((N, 1)), axis=1)

    w = w0 if w0 else np.zeros((C, D))    
    b = b0 if b0 else np.zeros(C)
    
    #create a new W-b combination so that so
    new_w = np.append(w, np.array([b]).T, axis=1)
    
    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for i in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################	
            num = np.exp((new_w @ new_X[n].T) - np.amax(new_w @ new_X[n].T))
            den = np.sum(num)
            P = num/den
            P[y[n]] -= 1
            new_w = new_w - (step_size) * np.array([P]).T @ np.array([new_X[n]])		
        
        

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################
        for i in range(max_iterations):
            num = np.exp((new_X @ new_w.T) - np.amax(new_X @ new_w.T))
            den = np.sum(num, axis=1)
            new_w = new_w - (step_size / N) * (((num.T / den).T - np.eye(C)[y]).T @ new_X)

    else:
        raise "Undefined algorithm."

    assert w.shape == (C, D)
    assert b.shape == (C,)
    
    # extract W from the W-b combination, access the b from W-b Combination
    return np.delete(new_w, -1, 1), new_w[:, -1]


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################
    w = np.insert(w, 0, b, axis=1).T
    X = np.insert(X, 0, 1, axis=1)
    return np.argmax(X@w, axis=1)
    
    #assert preds.shape == (N,)
    #return preds
