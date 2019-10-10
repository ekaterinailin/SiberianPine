import numpy as np

from ..utils import (generate_random_power_law_distribution,
                     logit,
                     )

def test_generate_random_power_law_distribution():
    pass
    
def test_logit():
    '''Test logit function by logifying
    some random function'''

    def func(x,a,b):
        return x / (b - a)
    log_uninf_prior = logit(func)
    up = func(3,2,6)
    logup = log_uninf_prior(3,2,6)
    assert np.log(0.75) == logup
