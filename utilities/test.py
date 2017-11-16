# import the packages
import numpy as np
from scipy.optimize import minimize
import scipy.stats as stats
import time

# Set up your x values
x = np.linspace(0, 100, num=100)

# Set up your observed y values with a known slope (2.4), intercept (5), and sd (4)
yObs = 5 + 2.4*x + np.random.normal(0, 4, 100)

# Define the likelihood function where params is a list of initial parameter estimates
def regressLL(params):
    # Resave the initial parameter guesses
    b0 = params[0]
    b1 = params[1]
    sd = params[2]

    # Calculate the predicted values from the initial parameter guesses
    yPred = b0 + b1*x

    # Calculate the negative log-likelihood as the negative sum of the log of a normal
    # PDF where the observed values are normally distributed around the mean (yPred)
    # with a standard deviation of sd
    logLik = -np.sum( stats.norm.logpdf(yObs, loc=yPred, scale=sd) )

    # Tell the function to return the NLL (this is what will be minimized)
    return(logLik)

# Make a list of initial parameter guesses (b0, b1, sd)    
initParams = [1, 1, 1]

# Run the minimizer
results = minimize(regressLL, initParams, method='nelder-mead')

# Print the results. They should be really close to your actual values
print(results.x)