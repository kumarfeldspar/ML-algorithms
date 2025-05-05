# Regression

This repository contains a Python implementation of linear regression, logistic regression, and ridge regression algorithms. These algorithms are commonly used in machine learning and statistical modeling for various tasks such as predicting numerical values, classifying data into categories, and handling multicollinearity in regression models. The implementation includes L2 regularization, also known as ridge regression, to mitigate overfitting.

## Files

The repository consists of the following files:

1. `linreg.py`: This file contains the implementation of the regression algorithms as well as utility functions for data preprocessing and optimization.

## Usage

To use the regression algorithms, follow the instructions below:

1. Ensure that you have the necessary dependencies installed. The code requires `numpy` and `pandas` libraries.

2. Copy the `linreg.py` file into your project directory.

3. Import the desired regression class (`LinearRegression621`, `LogisticRegression621`, or `RidgeRegression621`) into your Python script:

```python
from linreg import LinearRegression621, LogisticRegression621, RidgeRegression621
```

4. Create an instance of the regression class, optionally setting the desired hyperparameters:

```python
regressor = LinearRegression621(eta=0.00001, lmbda=0.0, max_iter=1000)
```

5. Fit the regression model to your data:

```python
regressor.fit(X, y)
```

6. Predict using the trained model:

```python
predictions = regressor.predict(X_test)
```

## Linear Regression without Regularization
Linear regression is a widely used algorithm for predicting numerical values based on input features. In its basic form, linear regression aims to minimize the sum of squared residuals between the predicted and actual values. This can be achieved by finding the optimal values of the coefficient vector β that minimizes the loss function.

In the context of linear regression without regularization, the loss function is defined as:

```math
L(\beta) = (y - X\beta)^T(y - X\beta)
```

where β is the coefficient vector and X is the design matrix. The gradient of the loss function with respect to β is calculated as:

```math
\nabla L(\beta) = -X^T(y - X\beta)
```

To train a linear regression model without regularization, the implementation in this repository uses gradient descent to iteratively update the coefficient vector β. The goal is to find the optimal values of β that minimize the loss function and provide the best fit to the data.

## L2 Regularization in Linear Regression (Ridge Regression)

In addition to the basic linear regression and logistic regression algorithms, this repository includes an implementation of ridge regression, which employs L2 regularization. L2 regularization is a technique used to prevent overfitting in regression models adding a penalty term to the loss function. That is, the loss function is modified to minimize the sum of squared residuals plus λ∑(βⱼ)², where λ is the regularization parameter.

The ridge regression implementation (`RidgeRegression621`) in this repository allows you to control the amount of regularization by specifying the `lmbda` parameter. A higher value of `lmbda` leads to stronger regularization and can help reduce the impact of multicollinearity in the data.

When using ridge regression, it's important to note that the bias term (intercept) β₀ is not regularized. The regularization is only applied to the coefficients of the independent variables.

## Logistic Regression without Regularization

For logistic regression without regularization, a different loss function is used, namely the negative log-likelihood. In this case, a column of 1s is  added to the design matrix X to estimate the bias term (intercept). The gradient of the negative log-likelihood loss function is calculated as:

```math
\nabla L(\beta) = -X^T(y - sigmoid(X\beta))
```

## Adagrad Optimization

The implementation in this repository utilizes the Adagrad version of gradient descent for optimization. The Adagrad algorithm adapts the learning rate for each parameter based on the historical gradient information. The steps involved in the Adagrad optimization algorithm are as follows:

```
minimize(X, y, ∇L, η, ε=1e-5, precision=1e-9)
    B ∼ 2N(0, 1) - 1  # Initialize beta with random values between -1 and 1
    h = 0  # Initialize the sum of squared gradient history
    X = (1, X)  # Add a column of 1s to the design matrix
    while ||∇L(β)||₂ ≤ precision:
        h += sum of squared partial derivatives
        β = β - (η ∇L / sqrt(h + ε))
    return β
```

By using Adagrad, you can enhance the convergence speed of the gradient descent algorithm and potentially improve the optimization process.

