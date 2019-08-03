import pytest
import ron_gauss
import numpy as np

def test_data_preprocessing():
    test_data = np.load('./tests/test_data/test_data_preprocessing.npz')
    input = test_data['input']
    output = test_data['output']
    prng = np.random.RandomState(seed=7)
    
    x_bar, mu_dp = ron_gauss.RONGauss()._data_preprocessing(input, epsilon_mean=1.0, prng=prng)
    
    assert (output[0] == x_bar).all() and  (output[1] == mu_dp).all()
