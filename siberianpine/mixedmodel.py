import corner
import emcee
import numpy as np

class MixedModel(object):
    """Combine multiple FFDs and fit 
    their parameters simultaneously with
    shared alpha.
    """
    def __init__(self, BFA=[], loglikelihood=None, alpha_prior=None):
        '''Constructor for a Mixed Model Bayesian analysis suite. 
        NOT TESTED.
        
        Attributes:
        -----------
        BFA : list of BayesianFlaringAnalysis objects
        
        loglikelihood : func
            loglikelihood function
        alpha_prior : float
            shared prior for alpha
        '''
        self.BFA = BFA
        self.loglikelihood = loglikelihood
        self.alpha_prior = alpha_prior
        
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

        args, inits = [], []
        
        for bfa in BFA:
            args.append([bfa.mined, bfa.Tprime, bfa.Mprime,
                         bfa.deltaT, bfa.threshed, bfa.M,
                         bfa.events])
            inits.append(bfa.eps_prior)
            
        inits.append(self.alpha_prior)    
        
        args = [i for i in args if i is not None]
        inits = [i for i in inits if i]
        
        ndim = len(inits)
        pos = [inits + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.loglikelihood, args=args)
        sampler.run_mcmc(pos, steps)

        self.samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))

    def show_corner_plot(self, save=False, path=''):
        '''Show (and save) a corner plot. NOT TESTED.

        '''
        truths = [bfa.eps_prior for bfa in BFA]
        truths.append(2.)

        ndim = len(BFA)
        labels = [r'$\epsilon_{}$'.format(i) for i in range(ndim)] + [r'$\alpha$']
        fig = corner.corner(MM.samples, 
                            labels=labels,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True,
                            title_kwargs={"fontsize": 12},
                            truths=truths,)
        if save==True:
            fig.savefig(path, dpi=300)
            
