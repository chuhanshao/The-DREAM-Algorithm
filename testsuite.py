import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as ss
import math
import random
from collections import namedtuple

def calculate_optimal_sigma(dim):
    return 2.38/np.sqrt(dim)

def state_space(dim, max_N=10000):
    return {'dim': dim,
            'Origin': np.zeros(dim),
            'Id': np.eye(dim),
            'sigma_opt':calculate_optimal_sigma(dim),
            'max_N':max_N}

def generate_random_state(sp, min_range=-10, max_range=10):
    """Generates a random state in the state space that fits in the area to be plotted.
    """
    return np.random.uniform(low=min_range, high=max_range, size=sp['dim'])

def generate_initial_states(sp, nb_runs):
    initial_states = {i:generate_random_state(sp) for i in np.arange(nb_runs)}
    # Only update if the key does not exist yet. Check out how to do this.
    sp.update({'Initial States':initial_states})

def generate_rotation_matrix(theta):
    # Rotation matrix is 2-dimensional
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def generate_correlated_cov(uncorrelated_cov, theta):
    correlated_cov = np.copy(uncorrelated_cov)
    R = generate_rotation_matrix(theta)
    R_inv = la.inv(R)
    # Rotate the first 2 dimensions only and leave the other dimensions
    # of the covariance matrix intact.
    correlated_cov[:2, :2] = R @ uncorrelated_cov[:2,:2] @ R_inv
    return correlated_cov

def get_ellipse_parameters(cov):

    """Get the first 2 eigenvalues and their angle of covariance matrix.
    The eigenvalues are returned in descending order together with
    the angle of rotation (in radians). The eigenvalues correspond with
    half the length, a and b, of these two main axes of
    the general ellipse.
    If the angle is small enough, meaning that the covariance matrix
    can be considered diagonal, 0.0 is returned."""

    e, v = la.eig(cov)
    e_1, e_2, *_ = e
    a, b = np.sqrt(e_1), np.sqrt(e_2)
    v_a, v_b, *_ = v
    # a must be at least b
    if a < b:
        a, b = b, a
        v_a, v_b = v_b, v_a
    cos, *_ = v_a
    theta = np.arccos(cos)
    if np.isclose(theta, 0):
        theta = 0.0
    return a, b, theta

def calculate_ellipse_coefficients(a, b, theta):
    sin, cos = np.sin(theta), np.cos(theta)
    cos_sqd, sin_sqd = cos**2, sin**2
    a_sqd, b_sqd = a**2, b**2
    A = cos_sqd/a_sqd + sin_sqd/b_sqd
    C = sin_sqd/a_sqd + cos_sqd/b_sqd
    B = (1/a_sqd - 1/b_sqd)*sin*cos
    return A, B, C

def get_Gaussian_contour(cov):
    a, b, theta = get_ellipse_parameters(cov)
    A, B, C = calculate_ellipse_coefficients(a, b, theta)
    return lambda x1, x2: A*x1**2 + 2*B*x1*x2 + C*x2**2

def get_Gaussian_contour_2(cov, i, j):
    '''Gets a covariance matrix cov of arbitrary dimension and two integers i and j.
    It returns a 2-dimensional quadratic form with symmetric matrix consisting of the elements
    prec[i, i], prec[i, j], prec[j, i], and prec[j,j]. Here, prec is the precision,
    i.e. the inverse of the covariance. This quadratic form is used to draw contour lines.
    The indices i and j have to be different and less than the dimension of the state space.'''
    precision = la.inv(cov)
    if j < i:
        i, j = j, i
    two_dim = precision[np.ix_([i, j],[i, j])]
    A, B, C = two_dim[0, 0], two_dim[0, 1], two_dim[1, 1]
    return lambda x1, x2: A*x1**2 + 2*B*x1*x2 + C*x2**2

def get_chi2s(df, confidence_levels=[0.67, 0.90, 0.95, 0.99]):
    """ppf stands for the percent point function (inverse of cdf — percentiles)."""
    #contour_levels = {conf:ss.chi2.ppf(conf, df) for conf in confidence_levels}
    contour_levels = [ss.chi2.ppf(conf, df) for conf in confidence_levels]
    return contour_levels

def generate_Gaussian(sp, name, mean, cov):
    d = sp['dim']
    rv = ss.multivariate_normal(mean=mean, cov=cov)
    return {'Name':name,
            'State Space':sp,
            'pdf':rv.pdf,
            'log pdf':rv.logpdf,
            'Mean':mean,
            'Covariance':cov,
            'Contour Levels':get_chi2s(df=2),
            'Fraction Levels':get_chi2s(df=d)
            #'Samples':None,
           }

def rescale(cov):
    '''Rescale the covariance so that its largest eigenvalue is 100 which
    is also the largest eigenvalue of the distributions Pi_1 and Pi_2 in the testsuite.'''
    D, B = la.eigh(cov)
    largest = np.max(D)
    scale = 100/largest
    rescaled_D = np.diag(scale*D)
    return B@rescaled_D@B.T

def generate_covs(sp):
    # Standard Normal Z has the identity matrix as covariance
    identity = sp['Id']

    # The optimal isotropic proposal is $\sigma_{opt} * Id$
    var_opt = sp['sigma_opt']**2
    cov_prop = var_opt*identity

    # P1_2
    cov_Pi_1 = np.copy(identity)
    cov_Pi_1[0, 0] = 100

    # Pi_2
    cov_Pi_2 = generate_correlated_cov(cov_Pi_1, np.pi/4)

    # Pi_rnd
    d = sp['dim']
    M = np.random.normal(loc=0.0, scale=1.0, size=(d,d))
    cov_rnd = M@M.T

    return {'Z':identity, 'Proposal':cov_prop, 'Pi_1':cov_Pi_1,
            'Pi_2':cov_Pi_2, 'Pi_rnd':rescale(cov_rnd)}


def generate_all_Gaussians(sp):
    named_covs = generate_covs(sp)
    gaussians = {name:generate_Gaussian(sp=sp, name=name, mean=sp['Origin'], cov=cov)
                 for name, cov in named_covs.items()}
    return gaussians

def generate_isotropic_Gaussian(sp, sigma):
    origin, identity = sp['Origin'], sp['Id']
    diagonal = sigma**2 * identity
    return generate_Gaussian(sp=sp, name='Isotropic', mean=origin, cov=diagonal)

def generate_random_Gaussian(sp):
    d, origin = sp['dim'], sp['Origin']
    M = np.random.normal(size=(d,d))
    random_cov = M@M.T
    return generate_Gaussian(sp=sp, name='Random', mean=origin, cov=random_cov)

def f_twist(b):
    def phi_b(x):
        """Argument and the value returned are d-dimensional numpy arrays."""
        y = np.copy(x)
        x1, x2 = x[:2]
        y[0], y[1] = x1, x2 + b*x1**2 - 100*b
        return y

    def phi_b_inv(y):
        """Argument and the value returned are d-dimensional numpy arrays."""
        x = np.copy(y)
        y1, y2 = y[:2]
        x[0], x[1] = y1, y2 - b*y1**2 + 100*b
        return x
    return phi_b, phi_b_inv

def compose2(f, g):
    return lambda x: f(g(x))

def apply_to(transformation, pts):
    """Used to generate samples of a twist distribution given samples of a Gaussian one.
    The argument transformation, e.g. phi_b(x1, x2) = (y1, y2) is a 2-dimensional
    transformation of the vectors in pts. The result is an array of the transformed points.
    """
    transformed_pts = np.zeros_like(pts)
    for i, pt in enumerate(pts):
        transformed_pts[i] = transformation(pt)
    return transformed_pts

def apply(transformation):
    return lambda pts: apply_to(transformation, pts)

def get_twisted_contour(gaussian, b):
    cov = gaussian['Covariance']
    f = get_Gaussian_contour(cov)
    return lambda x1, x2: f(x1, x2 + b*x1**2 - 100*b)

def generate_twist(gaussian, b, name):
    # The twisted distribution is a transformation of
    # the uncorrelated Gaussian distribution 'gaussian'
    transformed_distr = gaussian.copy()
    transformed_function, inverse_twist_function = f_twist(b=b)
    transformed_pdf = compose2(gaussian['pdf'], transformed_function)
    contour_function = get_twisted_contour(gaussian=gaussian, b=b)
    transformed_distr.update({'Name':name,
                              'Generator':gaussian,
                              'pdf':transformed_pdf,
                              'Contour Function':contour_function})
    transformed_distr.update({'Transformation':apply(inverse_twist_function)})
    return transformed_distr

def generate_all_twists(gaussian, b_values, names):
    twists ={name:generate_twist(gaussian, b, name)
             for b, name in zip(b_values, names)}
    return twists

def generate_test_suite(sp):
    gaussians = generate_all_Gaussians(sp)
    twists = generate_all_twists(gaussian=gaussians['Pi_1'],
                                 b_values=[0.03, 0.1],
                                 names=['Pi_3', 'Pi_4'])
    sp.update({'Test Suite':{**gaussians, **twists}})

def generate_state_space(dim, nb_runs=100, N=None):
    sp = state_space(dim=dim)
    generate_test_suite(sp)
    generate_initial_states(sp=sp, nb_runs=nb_runs)
    return sp

def iid_samples_Gaussian(gaussian, N):
    mean, cov = gaussian['Mean'], gaussian['Covariance']
    rv = ss.multivariate_normal(mean=mean, cov=cov)
    samples = rv.rvs(size=N)
    gaussian.update({'Samples':samples})

def iid_samples_transformed_Gaussian(distr, N):
    #Samples are generated by transforming the random samples of
    #the generating Gaussian distribution.
    generator = distr['Generator']
    transformation = distr['Transformation']
    if not 'Samples' in generator:
        iid_samples_Gaussian(generator, N)
    transformed_samples = transformation(generator['Samples'])
    distr.update({'Samples':transformed_samples})

def generate_iid_samples(sp, N):
    test_suite = sp['Test Suite']
    for name, distr in test_suite.items():
        if 'Generator' not in distr:
            iid_samples_Gaussian(gaussian=distr, N=N)
        else:
            iid_samples_transformed_Gaussian(distr=distr, N=N)

def get_distribution(sp, name):
    return sp['Test Suite'][name]

def get_samples(sp, name):
    return get_distribution(sp, name)['Samples']

def Mahalanobis_distance(mean, point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    delta = mean - point
    return np.sqrt(delta @ precision @ delta.T)

def squared_Mahalanobis_distance(point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    delta = mean - point
    return delta @ precision @ delta.T

def Mahalanobis_norm(point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    return np.sqrt(point @ precision @ point.T)

def squared_Mahalanobis_norm(point, precision):
    # The precision matrix is the inverse of the covariance matrix.
    return point @ precision @ point.T

def get_burnin_end(burnin_pct, nb_samples):
    return burnin_pct*nb_samples//100

def calculate_fractions(distribution, samples, burnin_pct=50):
    precision = la.inv(distribution['Covariance'])
    burnin_end = get_burnin_end(burnin_pct, len(samples))
    samples_after_burnin = samples[burnin_end:]
    nb_samples = len(samples_after_burnin)
    norm_sq = [squared_Mahalanobis_norm(sample, precision)
               for sample in samples_after_burnin]
    return [sum(norm_sq <= contour_level)/nb_samples
            for contour_level in distribution['Fraction Levels']]