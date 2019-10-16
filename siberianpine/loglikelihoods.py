import numpy as np

from .utils import logit

def mixedmodel_loglikelihood(theta, *args, prior=None):
    '''Custom likelihood to pass to 
    MixedModel with a prior of your choice.
    NOT TESTED
    
    theta : list of length n
        theta[:-2] eps_prior values
        theta[-1] shared alpha_prior
    args : list of lists 
        with the parameters for
        :func:`calculate_joint_posterior_distribution`
    prior : func
        prior function of your choice
    '''
    posterior = 0
    for i, beta in enumerate(theta[:-1]):
        # add the loglikelihoods
        posterior += calculate_joint_posterior_distribution([beta, theta[-1]], *args[i], prior)
    return posterior

def calculate_posterior_value_that_can_be_passed_to_mcmc(lp):
    '''Do some checks to make sure MCMC will work.'''
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    else:
        return lp


def calculate_joint_posterior_distribution(theta, mined, Tprime,
                                           Mprime, deltaT, threshed,
                                           M, events, prior):
    '''Equation (24) in Wheatland 2004.
    Log-probability distribution of an event bigger than mined
    occurring in an interval deltaT. 
    
    If there are no events observed during a campaign, we only update the
    flaring probability and fix alpha to the initial value with Eq. (23).

    Parameters:
    ----------
    theta : (float, float)
        probability, alpha
    mined : float
        minimum ED value to predict
    Tprime : float
        start time of relevant time period
    Mprime : int
        number of flares in relevant time period
    deltaT : float
        time interval after last observation
        to consider for predictions.
    threshed : float
        detection threshold for flare ED in s
    M : int
        same as Mprime unless you split your time series
        using bayesian blocks
    prior : func
        Prior distribution of alpha

    Return:
    -------
    probability distribution of at least one
    big event above mined during a period deltaT.
    '''
    np.seterr(divide='ignore')

    x, alpha = theta
    # This is the uniform prior for epsilon:
    if ((x < 0) | (x > 1)):
        return -np.inf
    else:
        # logify factors and add:
        if not (isinstance(events, np.ndarray) | isinstance(events, list)):
        
            raise ValueError("Flare event data must be a 1D numpy array or list.")
        
        events = np.array(events)
        # f1-f5 are factors in (24):
        f1 = Mprime * np.log(-np.log(1. - x))

        f2 = M * np.log(alpha - 1)

        # prior distribution for alpha:
        f3 = logit(prior)(alpha)

        _f4 = (Mprime + 1.) * np.log(mined / threshed)
        PI = np.sum(np.log(events / threshed))# eqn (13)
        f4 = alpha * (_f4 - PI)


        _f5 = Tprime / deltaT * np.power(mined / threshed, alpha - 1.) - 1.
        f5 =  _f5 * np.log(1. - x)


       
        if len(events) > 0:
     
            lp = f1 + f2 + f3 + f4 + f5
            
        elif len(events) == 0:
            
            lp = f1 + f5 # if no events are observed default to Eq. (24)
        
         
        # Check for bad values before returning the result:

        return calculate_posterior_value_that_can_be_passed_to_mcmc(lp)
        

def occurrence_probability_posterior(x, alpha, mined, Tprime,
                                     Mprime, deltaT, threshed,):
    '''Equation (25) in Wheatland 2004.
    Probability distribution of an event bigger than mined
    occurring in an interval deltaT.

    Parameters:
    ----------
    x : array or float
        probability range (0,1) like np.linspace(1e-8,1-1e-8,N)
        with N = grid size.
    alpha : float
        power law exponent
    mined : float
        minimum ED value to predict
    Tprime : float
        start time of relevant time period
    Mprime : int
        number of flares in relevant time period
    deltaT : float
        time interval after last observation
        to consider for predictions.
    threshed : float
        detection threshold for flare ED in s

    Return:
    -------
    probability distribution of at least one
    big event above mined during a period deltaT.
    '''
    f1 = Mprime * np.log(-np.log(1. - x))
    f2 = np.log(1. - x) * ((Tprime / deltaT)
          * np.power(mined / threshed, alpha - 1.)
          - 1.)
    dist = f1 + f2
    return dist


def flaring_rate_likelihood(rates, Mprime, Tprime, norm=False):
    '''Equation (18) from Wheatland.
    Likelihood distribution for rates.

    Parameters:
    -----------
    rates : array or float
        flaring rates to consider in pdf
    Mprime : int
        number of flares in relevant time period
    Tprime : float
        start time of relevant time period
    norm : bool
        If True will normalize maximum
        likelihood distribution

    Return:
    -------
    Posterior distribution for rates.
    The probability for rates of flares with
    flaresprop properties being true given the data
    captured in Mprime and Tprime.
    '''
    likelihood = np.power(rates, Mprime) * np.exp(-rates * Tprime)
    if norm==True:
        norma = np.sum(likelihood)
        return likelihood / norma
    else:
        return likelihood
