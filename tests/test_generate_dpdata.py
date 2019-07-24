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


def test_dimensionality_agnostic_execution():
    for i in np.arange(2, 101):
        rand_num_pts = np.random.randint(100) + 100
        # Generate a random number of normally distributed data points in
        # the range of 101-200 in i dimensions.
        test_data = data_gen_helper(rand_num_pts, i)
        # Simulate DP-safe lower dimensionality data.
        test_ron_gauss_instance = ron_gauss.RONGauss(algorithm="unsupervised")
        dims_reduced = int(np.ceil((i - 1) * (rand_num_pts / 200)))
        dp_data = test_ron_gauss_instance.generate_dpdata(test_data, dims_reduced)
        # Test dimensions are as expected.
        assert dp_data[0].shape[0] == rand_num_pts
        assert dp_data[0].shape[1] == dims_reduced

        # Test that the reconstruct parameter is working regardless of projection dimension
        dp_data = test_ron_gauss_instance.generate_dpdata(test_data,
                                                          dims_reduced,
                                                          reconstruct=True)
        # Test dimensions are as expected.
        assert dp_data[0].shape[0] == rand_num_pts
        assert dp_data[0].shape[1] == i


def test_simple_unsupervised_data():
    test_data = data_gen_helper(100, 2)
    test_ron_gauss_instance = ron_gauss.RONGauss(algorithm="unsupervised")
    dp_data = test_ron_gauss_instance.generate_dpdata(test_data, 1)
    assert normality_test_helper(dp_data)


def test_simple_gmm_data():
    test_data = data_gen_helper(100, 2)
    test_labels = np.random.choice([0, 1], size=100, p=[1./2, 1./2])
    test_ron_gauss_instance = ron_gauss.RONGauss(algorithm="gmm")
    dp_data = test_ron_gauss_instance.generate_dpdata(test_data, 1, test_labels)
    assert normality_test_helper(dp_data)


def test_simple_supervised_data():
    test_data = data_gen_helper(1000, 3)
    test_labels = np.random.choice([0, 1], size=1000, p=[1./2, 1./2])
    test_ron_gauss_instance = ron_gauss.RONGauss(algorithm="supervised")
    # TODO: Specify a reasonable value for maxY that excercises the code in supervised condition.
    dp_data = test_ron_gauss_instance.generate_dpdata(X=test_data,
                                                      dimension=2,
                                                      y=test_labels,
                                                      maxY=np.array(1)
                                                      )
    assert normality_test_helper(dp_data)
