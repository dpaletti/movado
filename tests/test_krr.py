import random
from typing import List

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

######################################################################################################
from movado import approximate

X: List[float] = []
y: List[float] = []


def test_krr():
    global X
    global y
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
    x1, x2, f = generate_data(-10, 10, 1.0, 2)
    # Prepare X and y for KRR
    X_local, y_local = prepare_data_to_KRR(x1, x2, f)
    X = X_local
    y = y_local
    # Set limits for Differential Evolution
    KRR_alpha_lim = (0.00001, 100.0)
    KRR_gamma_lim = (0.00001, 20.0)
    boundaries = [KRR_alpha_lim] + [KRR_gamma_lim]
    # Set up Differential Evolution solver
    solver = differential_evolution(
        KRR_function,
        boundaries,
        strategy="best1bin",
        popsize=15,
        mutation=0.5,
        recombination=0.7,
        tol=0.01,
        seed=0,
    )
    # Calculate best hyperparameters and resulting rmse
    best_hyperparams = solver.x
    best_rmse = solver.fun
    # Print final results
    print(
        "Converged hyperparameters: alpha= %.6f, gamma= %.6f"
        % (best_hyperparams[0], best_hyperparams[1])
    )
    print("Minimum rmse: %.6f" % (best_rmse))


######################################################################################################
def generate_data(xmin, xmax, Delta, noise):
    # Calculate f=sin(x1)+cos(x2)
    x1 = np.arange(xmin, xmax + Delta, Delta)  # generate x1 values from xmin to xmax
    x2 = np.arange(xmin, xmax + Delta, Delta)  # generate x2 values from xmin to xmax
    x1, x2 = np.meshgrid(x1, x2)  # make x1,x2 grid of points
    f = np.sin(x1) + np.cos(x2)  # calculate for all (x1,x2) grid
    # Add random noise to f
    random.seed(2020)  # set random seed for reproducibility
    for i in range(len(f)):
        for j in range(len(f[0])):
            f[i][j] = f[i][j] + random.uniform(
                -noise, noise
            )  # add random noise to f(x1,x2)
    return x1, x2, f


######################################################################################################
def prepare_data_to_KRR(x1, x2, f):
    # Transform X into a 2D numpy array, and y into a 1D numpy array
    X = []
    for i in range(len(f)):
        for j in range(len(f)):
            X_term = []
            X_term.append(x1[i][j])
            X_term.append(x2[i][j])
            X.append(X_term)
    y = f.flatten()
    X = np.array(X)
    y = np.array(y)
    return X, y


######################################################################################################


def KRR_function(hyperparams):
    global X
    global y
    # Assign hyper-parameters
    alpha_value = hyperparams[0]
    gamma_value = hyperparams[1]
    # Split data into test and train: random state fixed for reproducibility
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    y_pred_total = []
    y_test_total = []
    # kf-fold cross-validation loop
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Scale X0train and X_test
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Fit KRR with (X_train_scaled, y_train), and predict X_test_scaled
        KRR = KernelRidge(kernel="rbf", alpha=alpha_value, gamma=gamma_value)
        y_pred = KRR.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # Append y_pred and y_test values of this k-fold step to list with total values
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)
    # Flatten lists with test and predicted values
    y_pred_total = [item for sublist in y_pred_total for item in sublist]
    y_test_total = [item for sublist in y_test_total for item in sublist]
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    print(
        "alpha: %.6f . gamma: %.6f . rmse: %.6f" % (alpha_value, gamma_value, rmse)
    )  # Uncomment to print intermediate results
    return rmse


test_krr()
