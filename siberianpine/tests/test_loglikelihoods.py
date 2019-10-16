import numpy as np
import pytest
from ..loglikelihoods import (flaring_rate_likelihood,
                              occurrence_probability_posterior,
                              calculate_joint_posterior_distribution,
                              calculate_posterior_value_that_can_be_passed_to_mcmc,
                              mixedmodel_loglikelihood)
from ..utils import generate_random_power_law_distribution, generate_synthetic_bfa_input
from ..priors import uninformative_prior


def test_mixedmodel_loglikelihood():
    # ------------------------------------------------
    # Create some data 
    theta = (.1,.1, 2.)
    # args = [mined, Tprime, Mprime, deltaT, threshed, M, events]
    argsc = [20, 2, 3, 20, 1, 3,[10,20,30]] 
    args = [argsc, argsc]

    # Test one functional case
    def prior(x):
        return uninformative_prior(x,1.8,2.2)
        
    assert mixedmodel_loglikelihood(theta, *args, prior=prior) == pytest.approx(1.2533138525810665)

    # Test a case with a higher likelihood 
    def prior(x):
        return uninformative_prior(x,1.98,2.02)

        assert mixedmodel_loglikelihood(theta, *args, prior=prior) == pytest.approx(5.858484038569156)

    # Remove the events from one data set
    args[1][-1] = []
    args[1][2] = 0
    args[1][-2] = 0

    assert mixedmodel_loglikelihood(theta, *args, prior=prior) == pytest.approx(-0.21072103131565256
    )

    # Remove the events from the other
    args[0][-1] = []
    args[0][2] = 0
    args[0][-2] = 0
    assert mixedmodel_loglikelihood(theta, *args, prior=prior) == pytest.approx(-0.21072103131565256)

    # Choose eps out of range (0,1)
    args = [argsc, argsc]
    theta = (-0.02,.1, 2.)
    assert mixedmodel_loglikelihood(theta, *args, prior=prior) == -np.inf

    # Wrong data formats in one sample are enough to let the entire process fail as it should
    args[0][-1] = 5.
    with pytest.raises(ValueError) as err:
        mixedmodel_loglikelihood(theta, *args, prior=prior)


def test_calculate_posterior_value_that_can_be_passed_to_mcmc():
    #Just test the diffent cases:
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(4) == 4.
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(0.) == 0.
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(np.inf) == -np.inf
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(-np.inf) == -np.inf
    assert calculate_posterior_value_that_can_be_passed_to_mcmc(np.nan) == -np.inf
    
def test_calculate_joint_posterior_distribution():
    
    # Test some real values
    #------------------------------------------------------------------------------

    def prior(x):
        return uninformative_prior(x,1.8,2.2)

    bfa = generate_synthetic_bfa_input(seed=20)
    theta = (bfa["eps_prior"], bfa["alpha_prior"])


    res = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                               bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                               bfa["Tprime"], bfa["events"], prior)


    assert isinstance(res, float)
    assert res == pytest.approx(1828.9078245805415)

    # Test some real values
    #------------------------------------------------------------------------------
    bfa = generate_synthetic_bfa_input(flares_per_day=0.001, seed=1003)

    theta = (bfa["eps_prior"], bfa["alpha_prior"])

    resw = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                               bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                               bfa["Tprime"], bfa["events"], prior)
    assert resw == pytest.approx(-0.04999)

    # 
    #------------------------------------------------------------------------------
    bfa = generate_synthetic_bfa_input(flares_per_day=0.0, seed=1003)

    theta = (bfa["eps_prior"], bfa["alpha_prior"])

    resw = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                               bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                               bfa["Tprime"], bfa["events"], prior)
    assert resw == -np.inf

    # Test wrong input events with error string
    #------------------------------------------------------------------------------
    err_string = ("Flare event data must be a 1D numpy array or list.")
    with pytest.raises(ValueError) as err:
        resw = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                           bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                           bfa["Tprime"], 60., prior)
        assert err_string == err.value.args[0]
        
    #------------------------------------------------------------------------------
    def prior(x):
        return uninformative_prior(x,1.98,2.02)

    bfa = generate_synthetic_bfa_input(flares_per_day=1., seed=22)
    theta = (bfa["eps_prior"], bfa["alpha_prior"])
    resw = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                           bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                           bfa["Tprime"], bfa["events"], prior)
    assert resw==pytest.approx(112.66378606364091)

    # Test how a more informative prior increases the likelihood value
    #------------------------------------------------------------------------------
    def prior(x):
        return uninformative_prior(x,1.999,2.0001)

    bfa = generate_synthetic_bfa_input(flares_per_day=1., seed=22)
    theta = (bfa["eps_prior"], bfa["alpha_prior"])
    resw = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                           bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                           bfa["Tprime"], bfa["events"], prior)
    assert resw==pytest.approx(116.25735533795043)


    # Test how a more informative prior increases the likelihood value
    #------------------------------------------------------------------------------
    def prior(x):
        return uninformative_prior(x,1.99999,2.000001)

    bfa = generate_synthetic_bfa_input(flares_per_day=1., seed=22)
    theta = (bfa["eps_prior"], bfa["alpha_prior"])
    resw = calculate_joint_posterior_distribution(theta, bfa["mined"], bfa["Tprime"],
                                           bfa["Mprime"], bfa["deltaT"], bfa["threshed"],
                                           bfa["Tprime"], bfa["events"], prior)

    assert resw==pytest.approx(120.86252552391994)



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

