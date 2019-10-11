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
    times : array
        times of events
    events : array
        energies of events
    mined : float
        energy at which the cumulative 
        flare frequency is evaluated
    threshed : float
        detection threshold for flares 
        in the sample (not very well defined
        because it is energy dependent here)
    deltaT : float
        time interval considered for prediction
        of flaring rate above mined
    Tprime : float
        total observation time (light curve length)
        in days
    alpha_prior : float
        prior on alpha (e.g. 2.0)
    eps_prior : float
        prior on flaring probability within delta T
        as in:
        eps_prior = 1 - np.exp(-rate_prior * deltaT)
        where rate_prior is the prior in flares per day
    Mprime : int
        total number of events
    M : int
        number of events in an interval that can
        be considered a Poisson process with a constant
        intensity.
    samples : ndarray
        MCMC sampling result
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

        # output data for post-processing or plotting
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







