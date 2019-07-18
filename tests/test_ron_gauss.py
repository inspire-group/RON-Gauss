import pytest
import ron_gauss
import numpy as np

def test_data_preprocessing():
    test_data = np.load('./tests/test_data/test_data_preprocessing.npz')
    input = test_data['input']
    output = test_data['output']

    rongauss = ron_gauss.RONGauss(algorithm="supervised", epsilon_mu=1.0, epsilon_sigma=1.0)
    x_bar, mu_dp = rongauss._data_preprocessing(input, epsMu=1.0, prng_seed=7)
    
    assert (output[0] == x_bar).all() and  (output[1] == mu_dp).all()



# if __name__ == "__main__":
#     test_data_preprocessing()