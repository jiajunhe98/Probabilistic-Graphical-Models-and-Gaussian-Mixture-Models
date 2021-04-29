import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dnorm(x, mu, sigma):
    sigma += np.eye(sigma.shape[0]) * 1e-8
    return ss.multivariate_normal.logpdf(x, mu, sigma).reshape(1, -1)


def posterior_dirichlet(alpha, z):
    return alpha + np.array([np.sum(z == i) for i in range(alpha.shape[0])])


def posterior_NIW(mu0, lambd, scale, df, data_in_cluster):

    n = data_in_cluster.shape[0]
    mean = np.mean(data_in_cluster, axis=0, keepdims=True) if n != 0 else None

    # Calculate Posterior NIW Distribution's parameters
    mu0_post = (lambd * mu0 + n * mean) / (lambd + n) if n != 0 else mu0
    lambd_post = lambd + n
    df_post = df + n
    scale_post = scale + np.dot((data_in_cluster - mean).T, (data_in_cluster - mean)) \
                 + lambd * n / lambd_post * np.dot((mean - mu0).T, (mean - mu0)) if n != 0 else scale

    return mu0_post, lambd_post, df_post, scale_post


def dGMM(x, pi, alpha, mus, sigmas, mu0, lambd, df, scale, z):
    log_prob = ss.dirichlet.logpdf(pi, alpha)
    log_prob += sum([ss.invwishart.logpdf(sigmas[cluster], df=df, scale=scale)
                     for cluster in range(len(sigmas))])
    log_prob += sum([ss.multivariate_normal.logpdf(mus[cluster], mu0, sigmas[cluster] / lambd)
                     for cluster in range(len(sigmas))])
    log_prob += sum(np.log(pi[z]))
    log_prob += sum([ss.multivariate_normal.logpdf(x[i], mus[z[i]], sigmas[z[i]])
                     for i in range(x.shape[0])])
    return log_prob


class GMM_DataAugmentation:
    """
    GMM by Data Augmentation Method.

    Methods:
        fit(data, n_samples, burning_time): Fit the model to data.
        predict(x): Predict cluster labels for x.
    """

    def __init__(self, n_clusters):
        """
        Constructor Methods:

        Args:
            n_clusters(int): number of clusters.
        """

        self.n_clusters = n_clusters

        self.pi = None
        self.mus = [None] * self.n_clusters
        self.sigmas = [None] * self.n_clusters

    def fit(self, data, n_samples=20, max_iter=50):

        """
        Fit the model to data.

        Args:
            data: Array-like, shape (n_samples, n_dim)
            n_samples(int): Amount of samples
        """

        assert data.ndim == 2
        n_dim = data.shape[1]
        n_data = data.shape[0]

        # Set prior distribution
        alpha = np.full((self.n_clusters, ), 2)     # Dirichlet Prior
        mu0, lambd, scale, df = np.zeros(n_dim), 1, np.eye(n_dim), n_dim     # NIW Prior

        # Initialize samples
        z_samples = [self._initialization(data)] * n_samples

        for iter in range(max_iter):

            # P-step:
            mu_samples = [[] for cluster in range(self.n_clusters)]
            sigma_samples = [[] for cluster in range(self.n_clusters)]
            pi_samples = []

            for i in range(n_samples):
                sample = np.random.randint(0, n_samples, 1).item()
                z_sample = z_samples[sample]
                for cluster in range(self.n_clusters):

                    mu0_post, lambd_post, df_post, scale_post = \
                        posterior_NIW(mu0, lambd, scale, df, data[z_sample == cluster])
                    sigma_samples[cluster].append(ss.invwishart.rvs(df=df_post, scale=scale_post, size=1))
                    mu_samples[cluster].append(
                        ss.multivariate_normal.rvs(mean=mu0_post.reshape(-1), cov=sigma_samples[cluster][-1]/lambd_post, size=1))
                alpha_post = posterior_dirichlet(alpha, z_sample)
                pi_samples.append(ss.dirichlet.rvs(alpha=alpha_post, size=1).reshape(-1))
            z_samples = []

            # I-step
            for sample in range(n_samples):
                pi_sample = pi_samples[sample]
                log_prob = [dnorm(data, mu_samples[cluster][sample], sigma_samples[cluster][sample]) + np.log(pi_sample[cluster])
                                for cluster in range(self.n_clusters)]
                log_prob = np.vstack(log_prob)
                log_prob -= np.max(log_prob, axis=0, keepdims=True)
                prob = np.exp(log_prob) + 1e-8
                prob /= np.sum(prob, axis=0, keepdims=True)
                z_sample = np.array(
                    [np.argwhere(ss.multinomial.rvs(p=prob[:, j], n=1, size=1).reshape(self.n_clusters) == 1).item()
                        for j in range(n_data)])
                z_samples.append(z_sample)

        P_MAP = -np.inf
        for sample in range(n_samples):
            mus_sample = [mu_samples[cluster][sample] for cluster in range(self.n_clusters)]
            sigmas_sample = [sigma_samples[cluster][sample] for cluster in range(self.n_clusters)]
            P = dGMM(data, pi_samples[sample], alpha, mus_sample, sigmas_sample, mu0, lambd, df, scale, z_samples[sample])
            if P > P_MAP:
                self.pi, self.mus, self.sigmas, P_MAP = pi_samples[sample], mus_sample, sigmas_sample, P

        return self

    def predict(self, x):
        """
        Predict cluster labels for x.

        Args:
            x: Array-like, shape (n_samples, n_dim)
        Return:
            Array-like, shape (n_samples, )
        """
        log_prob = [dnorm(x, self.mus[cluster], self.sigmas[cluster]) + np.log(self.pi[cluster])
                     for cluster in range(self.n_clusters)]
        log_prob = np.vstack(log_prob)
        z = np.argmax(log_prob, axis=0)
        return z

    def _initialization(self, data, max_iter=5):
        """
        Initialization by K-Means.
        """
        means = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]    # pick random samples as center
        z = np.zeros(data.shape[0])

        for iter in range(max_iter):
            dist = [np.sum((data - means[cluster]) ** 2, axis=1) for cluster in range(self.n_clusters)]
            dist = np.vstack(dist)
            z = np.argmin(dist, axis=0)
            means = [np.mean(data[z == cluster], axis=0) for cluster in range(self.n_clusters)]

        return z







