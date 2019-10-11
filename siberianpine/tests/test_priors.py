import numpy as np
from ..priors import uninformative_prior

def test_uninformative_prior():
    '''Test some exceptions that should be mostly ignored.'''

    # Define test values:
    vals = [(1,2,3), (2,1,3), (np.nan, 4, 10), (3, 2, 2)]

    # Run prior calculation on values:
    res = []
    for rate, minrate, maxrate in vals:
        res.append(uninformative_prior(rate, minrate, maxrate))

    # Check results:
    assert res == [0,0.5,0,0]
