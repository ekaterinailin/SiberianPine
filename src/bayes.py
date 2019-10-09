# This code contains an analysis suite designed to predict
# flaring activity rates and energy distributions from (non-)observations
# of stellar flares.
#
# --------------------------------------------------------------------------
#
# Basic USAGE:
#
# # Read in likelihood functions, prior distributions:
#
#  >>> from opencluster.bayes import (calculate_joint_posterior_distribution,
#  >>>                                uninformative_prior, gaussian_prior)
#
# # Get the analysis toolkit:
#
#  >>> from opencluster.bayes import BayesianFlaringAnalysis
#
# # Mix and match priors with likelihoods into your final posterior:
#
#  >>> def loglikelihood(theta, *args):
#  >>>     '''Some posterior distribution.'''
#  >>>     ...
#  >>>     return ...
#
# # Initialize with all the relevant parameters:
#
#  >>> BFA = BayesianFlaringAnalysis(mined=3000, Tprime=1, deltaT=.4,
# #                                  alpha_prior=1.8, eps_prior=0.1,
# #                                  threshed=350, Mprime=0,
# #                                  loglikelihood=loglikelihood)
#
# # Run MCMC to sample the posterior distribution:
#
#  >>> BFA.sample_posterior_with_mcmc()
#
# # Plot the samples to show if there is any covariance between variables:
#
#  >>> BFA.show_corner_plot()
#
# # Since the distributions are not quite symmetric instead of
# # median and std we calculate 16, 50, 84 percentiles:
# # The first tuple represents eps, the second alpha:
#
#  >>> BFA.calculate_percentiles()
#
# ------------------------------------------------------------------------

import numpy as np
import emcee
import corner

class BayesianFlaringAnalysis(object):
    '''Analysis suite for flaring statistics.

    Attributes:
    -----------
    times
    '''

    def __init__(self, times=None, events=None, mined=None,
                 threshed=None, deltaT=None, alpha_prior=None,
                 eps_prior=None, Tprime=None, Mprime=None,
                 loglikelihood=None, M=None, samples=None):
        '''Init a Bayesian analysis suite. NOT TESTED.'''
        # general data, not needed yet, but should be eventually
        self.times = times
        self.events = events

        # attributes that can in principle be derived from general data
        # todo: implement setters and getters
        self.Tprime = Tprime
        self.Mprime = Mprime
        if M is not None:
            self.M = M
        else:
            self.M = Mprime

        # chosen parameters for analysis
        self.mined = mined
        self.threshed = threshed
        self.deltaT = deltaT

        # priors for analysis
        self.alpha_prior = alpha_prior
        self.eps_prior = eps_prior

        # log likelihood
        self.loglikelihood = loglikelihood

        # output data if you only want to the post-processing or plotting
        self.samples = samples

    def sample_posterior_with_mcmc(self, nwalkers=300, cutoff=100, steps=500):
        '''Sample from the posterior using MCMC.

        Parameters:
        -------------
        inits : list
            initial variable values in the correct order
            for lnprob
        lnprob : func
            posterior distribution that takes
            inits as first argument and *args as second
            to last arguments
        nwalkers : int
            number of walkers to run around the parameter space
        cutoff : int
            You do not want to use values in the beginning
            of the chain, so cut them off.
        steps : int
            How long to run the walk.

        Return:
        --------
        Sampling results as ndarray with dimensions like
        len(init) x ((steps - cutoff) * nwalkers)
        '''
        args = [self.mined, self.Tprime, self.Mprime,
                self.deltaT, self.threshed, self.M,
                self.events]
        inits = [self.eps_prior, self.alpha_prior]

        args = [i for i in args if i is not None]
        inits = [i for i in inits if i]
        ndim = len(inits)
        pos = [inits + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.loglikelihood, args=args)
        sampler.run_mcmc(pos, steps)

        self.samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))


    def show_corner_plot(self, labels=(r'$\epsilon$', r'$\alpha$'),
                         save=False, path=''):
        '''Show (and save) a corner plot. NOT TESTED.

        '''

        fig = corner.corner(self.samples, labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 12},
                            truths=(self.alpha_prior, self.eps_prior),
                            title="hello")

        # For flare we so far have two dimensions fixed
        ndim =2

        # Extract the axes
        axes = np.array(fig.axes).reshape((ndim, ndim))

        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(self.alpha_prior, color="g")
            ax.axvline(self.eps_prior, color="g")

        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(self.eps_prior, color="g")
                ax.axhline(self.alpha_prior, color="g")
                ax.plot(self.eps_prior, self.alpha_prior, "sg")

        if save == True:
            fig.savefig(path, dpi=300)
        return fig

    def calculate_percentiles(self):
        '''Calculate best fit value and its uncertainties.

        Parameters:
        ------------
        samples : ndarray
            # of variables x sample size

        Return:
        --------
        a tuple of 3-tuples with
        (median, upper_uncert, lower_uncert)
        each.
        '''
        map_of_results = map(lambda v: (v[1],  v[2] - v[1], v[1] - v[0]),
                               zip(*np.percentile(self.samples, [16, 50, 84], axis=0)))
        percentiles = list(map_of_results)
        self.alpha_posterior = percentiles[1]
        self.eps_posterior = percentiles[0]
        return percentiles


def calculate_posterior_value_that_can_be_passed_to_mcmc(lp):
    '''Do some checks to make sure MCMC will work. NOT TESTED.'''
    if not np.isfinite(lp):
        return -np.inf
    if np.isnan(lp):
        return -np.inf
    else:
        return lp

def logit(function):
    '''Make a probability distribution
    a log probability distribution.'''
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        np.seterr(divide='ignore') # ignore division by zero because you want to have the -np.inf results
        result = np.log(result)
        return result
    return wrapper


def calculate_joint_posterior_distribution(theta, mined, Tprime,
                                           Mprime, deltaT, threshed,
                                           M, events, prior):
    '''Equation (24) in Wheatland 2004.
    Log-probability distribution of an event bigger than mined
    occurring in an interval deltaT. NOT TESTED.

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

        # logify factors and add:

        lp = f1 + f2 + f3 + f4 + f5

        # Check for bad values before returning the result:

        return calculate_posterior_value_that_can_be_passed_to_mcmc(lp)


def uninformative_prior(rate, minrate, maxrate):
    '''Uninformative prior for the rates.
    Uniform within [minrate, maxrate].

    Parameters:
    -------------
    rate : float

    minrate, maxrate : float
        interval in which rate is constrained

    Return:
        Prior probability
    '''
    if ((rate >= minrate) & (rate <= maxrate)):
        return 1. / (maxrate - minrate)
    else:
        return 0

def gaussian_prior(x, mu, sigma):
    '''Evaluate a normalized Gaussian function
    with mu and sigma at x. NOT TESTED.'''
    return  1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))


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
