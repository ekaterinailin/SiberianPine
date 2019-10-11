import numpy as np
import pytest
from ..loglikelihoods import (flaring_rate_likelihood,
                              occurrence_probability_posterior,
                              calculate_joint_posterior_distribution,)
from ..utils import generate_random_power_law_distribution


def test_calculate_joint_posterior_distribution():
    pass

def test_flaring_rate_likelihood():
    '''Test flaring rate likelihood with a
       5 flares per day rate.
    '''
    # Find the likelihood

    rates = np.linspace(1e-1, 20, 10000)
    posterior = flaring_rate_likelihood(rates, 75, 15., norm=True)

    #-----------------------------------------

    # Check some values:
    assert np.sum(posterior) == pytest.approx(1.)
    assert rates[np.argmax(posterior)] == pytest.approx(5., rel=1e-3)
    assert rates[np.argmin(np.abs(np.max(posterior)/2.-posterior))] ==  pytest.approx(4.351065, rel=1e-4)
    

def test_occurrence_probability_posterior():
    '''Test the posterior using a uniform prior'''

    # Simulate the results from Wheatland 2004 in Fig. 1:
    # Use their precise values:

    t = 5 #total observation time in days
    cadence = 4 #observations per hour
    obstimes = np.linspace(3000,3000+t,t*24*4) # 15 min cadence observations
    flaresperday = 5. # average flaring rate in flares per day
    np.random.seed(3000)
    times = obstimes[np.where(np.random.poisson(lam=1. / 24. / cadence * flaresperday, size=t*24*4))[0]]
    size = len(times)
    events = generate_random_power_law_distribution(1, 100000, -.8, size=size, seed=778)
    #time interval to consider for prediction
    Tprime = 5. # if bayesian blocks used: bins[-1] - bins[-2]
    mined = 100 # min ED value we want to predict a rate for, same as S2 in Wheatland paper
    deltaT = 1. # predict rate of flares above threshold for deltaT days in the futures
    alpha = 1.8 # fix power law exponent for now
    threshed = 1. # detection sensitivity limit
    # number of observations
    Mprime = size# if bayesian blocks used: values[-1]

    # Find the posterior distribution:

    x = np.linspace(1e-8,1-1e-8,10000)
    predicted_distr = occurrence_probability_posterior(x, alpha, mined, Tprime,
                                                       Mprime, deltaT, threshed)

    #--------------------------------------------------

    # Check some values:
    # TODO: use more restrictive check because you know how to seed numpy random generators now.

    assert x[np.argmax(predicted_distr)] > 0.110
    assert x[np.argmax(predicted_distr)] < 0.131
    assert x[np.argmin(np.abs(np.max(predicted_distr)/2.-predicted_distr))] ==  pytest.approx( 0.12741274, rel=1e-4)

    # --------------------------------------------------

    # If debugging is needed later:
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    # ax.plot(x, predicted_distr, color="r",
    #         label='ED={} s,  t={} d'.format(mined, deltaT))

    # ax.set_xlabel(r'$\varepsilon$: probability of one event'+'\nabove ED within t after last observation')
    # ax.set_ylabel(r'normalized $P_{\varepsilon}(\varepsilon)$')
    # ax.set_xlim(0,0.25)
    # ax.legend()

