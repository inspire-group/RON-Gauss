import ron_gauss
import numpy as np


def data_gen_helper(sz, ndim):
    mu = np.zeros(ndim)
    cov = np.cov(np.random.normal(size=(ndim, sz*10)))
    return np.random.multivariate_normal(mu, cov, sz)


def normality_test_helper(dpdata):
    # TODO: implement actual test of normality with appropriate sensitivity.
    return True


def test_defaults():
    test_ron_gauss_instance = ron_gauss.RONGauss()
    assert(test_ron_gauss_instance.algorithm == "supervised")
    assert(test_ron_gauss_instance.epsilonCov == 1.0)
    assert(test_ron_gauss_instance.epsilonMean == 1.0)


def test_simple_2d_gauss_data():
    test_data = data_gen_helper(100, 2)
    test_ron_gauss_instance = ron_gauss.RONGauss(algorithm="unsupervised")
    dp_data = test_ron_gauss_instance.generate_dpdata(test_data, 1)
    assert normality_test_helper(dp_data)
