import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    k     = opts['k']
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug   
    # Training
    mdl     = KNeighborsClassifier(n_neighbors = k)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred   = mdl.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
    error   = 1 - acc
    
    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        # cost  = alpha * error + beta * (num_feat / max_feat)
        cost  = alpha * error + beta * (num_feat / max_feat) * 10
    return cost


# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import confusion_matrix

# # 计算指标
# def calculate_metrics(y_true, y_pred):
#     tp = np.sum((y_true == 1) & (y_pred == 1))
#     tn = np.sum((y_true == 0) & (y_pred == 0))
#     fp = np.sum((y_true == 0) & (y_pred == 1))
#     fn = np.sum((y_true == 1) & (y_pred == 0))
    
#     acc = (tp + tn) / (tp + tn + fp + fn)
#     sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
#     precision = tp / (tp + fp) if (tp + fp) != 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
#     mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) != 0 else 0
    
#     return acc, sensitivity, precision, specificity, mcc

# # error rate
# def error_rate(xtrain, ytrain, x, opts):
#     # parameters
#     k     = opts['k']
#     fold  = opts['fold']
#     xt    = fold['xt']
#     yt    = fold['yt']
#     xv    = fold['xv']
#     yv    = fold['yv']
    
#     # Number of instances
#     num_train = np.size(xt, 0)
#     num_valid = np.size(xv, 0)
#     # Define selected features
#     xtrain  = xt[:, x == 1]
#     ytrain  = yt.reshape(num_train)  # Solve bug
#     xvalid  = xv[:, x == 1]
#     yvalid  = yv.reshape(num_valid)  # Solve bug   
#     # Training
#     mdl     = KNeighborsClassifier(n_neighbors = k)
#     mdl.fit(xtrain, ytrain)
#     # Prediction
#     ypred   = mdl.predict(xvalid)
#     acc     = np.sum(yvalid == ypred) / num_valid
#     error   = 1 - acc
    
#     return error, ypred

# # Error rate & Feature size
# def Fun(xtrain, ytrain, x, opts):
#     # Parameters
#     alpha    = 0.99
#     beta     = 1 - alpha
#     # Original feature size
#     max_feat = len(x)
#     # Number of selected features
#     num_feat = np.sum(x == 1)
#     # Solve if no feature selected
#     if num_feat == 0:
#         cost  = 1
#     else:
#         # Get error rate
#         error, ypred = error_rate(xtrain, ytrain, x, opts)
#         # Objective function
#         cost  = alpha * error + beta * (num_feat / max_feat)
        
#     # Calculate metrics
#     acc, sensitivity, precision, specificity, mcc = calculate_metrics(ytrain, ypred)
    
#     return cost, acc, sensitivity, precision, specificity, mcc

# # Example usage
# opts = {'k': 5, 'fold': {'xt': np.array([[1, 2], [3, 4]]), 'yt': np.array([0, 1]), 'xv': np.array([[5, 6]]), 'yv': np.array([1])}}
# x = np.array([1, 1])
# xtrain = np.array([[1, 2], [3, 4]])
# ytrain = np.array([0, 1])

# cost, acc, sensitivity, precision, specificity, mcc = Fun(xtrain, ytrain, x, opts)
# print(f"Cost: {cost}")
# print(f"Accuracy: {acc}")
# print(f"Sensitivity: {sensitivity}")
# print(f"Precision: {precision}")
# print(f"Specificity: {specificity}")
# print(f"MCC: {mcc}")