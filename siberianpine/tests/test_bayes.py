import pytest
import numpy as np
from ..utils import generate_random_power_law_distribution
from ..bayes import BayesianFlaringAnalysis
from ..loglikelihoods import calculate_joint_posterior_distribution
from ..priors import uninformative_prior
                    

def test_sample_posterior_with_mcmc():
    '''Integration test using the example
    in Wheatland (2004) and our own
    likelihoods and priors.
    '''

    # Create some fake data:

    t = 5 #total observation time in days, must be int
    cadence = 4 #observations per hour
    obstimes = np.linspace(3000,3000+t,t*24*cadence) # 15 min cadence observations
    flaresperday = 10. # average flaring rate in flares per day
    times = obstimes[np.where(np.random.poisson(lam=1. / 24. / cadence * flaresperday, size=t * 24 * cadence))[0]]
    size = len(times)
    alpha_prior = 1.8 # fix power law exponent for now
    events = generate_random_power_law_distribution(1, 1000, -alpha_prior + 1., size=size, seed=788)
    #time interval to consider for prediction
    Tprime = 5#np.max(times) -  np.min(times)# if bayesian blocks used: bins[-1] - bins[-2]
    mined = 100 # min ED value we want to predict a rate for, same as S2 in Wheatland paper
    deltaT = 1.# predict rate of flares above threshold for deltaT days in the futures
    threshed = 1 # detection sensitivity limit
    Mprime = size# if bayesian blocks used: values[-1]
    rate_prior = flaresperday / np.abs(alpha_prior - 1.) * np.power(mined, -alpha_prior +1.) # evaluate cumulative FFD fit at mined
    eps_prior = 1 - np.exp(-rate_prior * deltaT) # calculate poisson probability.  Eqn (5) in Wheatland (2004)

    #---------------------------------

    # Define your log-likelihood function:

    def loglikelihood(theta, *args):
        def prior(x):
            return uninformative_prior(x, 1.25, 2.25)
        return calculate_joint_posterior_distribution(theta, *args, prior)

    # Create a BayesianAnalysisObject

    BFA = BayesianFlaringAnalysis(mined=mined, Tprime=Tprime, deltaT=deltaT, alpha_prior=alpha_prior, eps_prior=eps_prior,
                              threshed=threshed, Mprime=Mprime, events=events, loglikelihood=loglikelihood)

    # Run MCMC to sample the posterior distribution

    BFA.sample_posterior_with_mcmc()

    #---------------------------------

    # Check that the function ran through

    assert BFA.samples.shape == (120000, 2) #this goes wrong if default values for steps/cutoff/nwalkers are changed
    assert BFA.samples[:,1].max() < 2.25 # as defined by prior on alpha
    assert BFA.samples[:,1].min() > 1.25 # as defined by prior on alpha
    assert BFA.samples[:,0].max() < 1 # eps is a probability
    assert BFA.samples[:,0].min() > 0 # eps is a probability

    # Check that the results are correct maybe?


def test_calculate_percentiles():
    '''Test if the percentiles come out
    correctly in a random gauss shaped sample.
    '''
    # Create some data with a seed:

    np.random.seed(seed=590)
    samples = np.array([np.random.normal(loc=1.0, scale=.2, size=1000),
                        np.random.normal(loc=2.0, scale=.4, size=1000)]).T
    BFA = BayesianFlaringAnalysis(samples=samples)

    #-----------------------------------

    # Run the function:

    pct = BFA.calculate_percentiles()

    #-----------------------------------

    # Check the results:

    assert pct[0] == (1.008030476417989, 0.19516752006028892, 0.21571271977738948)
    assert pct[1] == (1.980568570082061, 0.4199525752578359, 0.37256627621632066)
