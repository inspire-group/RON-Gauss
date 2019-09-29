import pytest
import ron_gauss
import numpy as np
from sklearn import preprocessing


def check_element_diff(x, y, tolerance=1e-15):
    return (np.abs(x - y) < tolerance).all()
    

def test_data_preprocessing():
    test_data = np.load('./tests/test_data/test_data_preprocessing.npz')
    input = test_data['input']
    output = test_data['output']
    prng = np.random.RandomState(seed=7)
    x_bar, mu_dp = ron_gauss.RONGauss()._data_preprocessing(input, epsilon_mean=1.0, prng=prng)
    assert check_element_diff(output[0],x_bar) and  check_element_diff(output[1], mu_dp)

def test_normalize_sample_wise():
    number_of_trials = 30
    for i in range(number_of_trials):
        test_data = np.random.normal(size=(100,10))
        benchmark = preprocessing.normalize(test_data)
        new_implementation = ron_gauss.RONGauss()._normalize_sample_wise(test_data)
        assert (np.abs(new_implementation-benchmark) < 1e-15).all()