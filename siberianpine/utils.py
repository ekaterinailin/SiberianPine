import numpy as np

def generate_random_power_law_distribution(a, b, g, size=1, seed=None):
    """
    Power-law generator for pdf(x)\propto x^{g-1} for a<=x<=b
    """
    if seed is not None:
        np.random.seed(seed)
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    return (ag + (bg - ag)*r)**(1./g)
    
def logit(function):
    '''Make a probability distribution
    a log probability distribution.'''
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        np.seterr(divide='ignore') # ignore division by zero because you want to have the -np.inf results
        result = np.log(result)
        return result
    return wrapper
    
def generate_synthetic_bfa_input(flares_per_day=10., 
                                 mined=100, Tprime=50, deltaT=1.,
                                 alpha_prior=2., threshed=1., cadence=4,
                                 t0=3000., seed=None, maxed=1e4,
                                 estimate_starting_points=False):
    """Generate a dictionary of inputs for 
    BayesianFlaringAnalysis.
    
    Parameters:
    -------------
    flares_per_day : float
        average flaring rate in flares per day
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
    cadence : int
        number of observations per hour
    t0 : float
        time offset
    seed : float
        seed the random generator with a number
        if needed
    maxed: float
        set a maximum value for ED, set >> mined
        to simulate a power law without coutoff
    estimate_starting_points : bool, default False
        If True will find MLE for alpha and eps to use as
        starting points for MCMC.

    """
    #time related stuff:
    size = int(np.rint(Tprime*24*cadence))
    obstimes = np.linspace(t0,t0+Tprime,size) # 15 min cadence observations

    times = obstimes[np.where(np.random.poisson(lam=1. / 24. / cadence * flares_per_day,
                                                size=size))[0]]
    Mprime = len(times) # number of events

    #energy related stuff

    # Generate power law distributed data:
    events = generate_random_power_law_distribution(1, maxed, -alpha_prior + 1., size=Mprime, seed=seed)
    threshed = 1 # detection sensitivity limit

    # determine a starting point for the MCMC sampling
    rate_prior = (flares_per_day / np.abs(alpha_prior - 1.) *
                  np.power(mined, -alpha_prior +1.)) # evaluate cumulative FFD fit at mined
    eps_prior = 1 - np.exp(-rate_prior * deltaT) #use Poisson process statistics do get a probability from the rate

    if estimate_starting_points==True:
        # For an uninfromative prior on alpha we expect the MCMC result to be:
        alpha_prior = Mprime / np.sum(np.log(events/threshed)) + 1.

        # determine a starting point for the MCMC sampling
        rate_prior = (flaresperday / np.abs(alpha_prior - 1.) *
                      np.power(mined, -alpha_prior +1.)) # evaluate cumulative FFD fit at mined
        eps_prior = 1 - np.exp(-rate_prior * deltaT) #use Poisson process statistics do get a probability from the rate

    return {"mined":mined, "Tprime":Tprime, "deltaT":deltaT,
            "alpha_prior":alpha_prior, "eps_prior":eps_prior,
            "threshed":threshed, "Mprime":Mprime, "events":events}
