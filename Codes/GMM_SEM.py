import numpy as np
import scipy.stats as ss


def dnorm(x, mu, sigma):
    sigma += np.eye(sigma.shape[0]) * 1e-8
    return ss.multivariate_normal.logpdf(x, mu, sigma)


class GMM_SEM:
    """
    GMM by stochastic EM.

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

    def fit(self, data, max_iter=2000, threshold=1e-8):
        """
        Fit the model to data.

        Args:
            data: Array-like, shape (n_samples, n_dim)
            max_iter(int)
        """

        assert data.ndim == 2
        n_data = data.shape[0]

        # Initialize
        z = self._initialization(data)
        self.pi = np.array([np.mean(z == cluster) for cluster in range(self.n_clusters)])
        self.mus = [np.mean(data[z == cluster, :], axis=0) for cluster in range(self.n_clusters)]
        self.sigmas = [np.cov(data[z == cluster, :].T) for cluster in range(self.n_clusters)]

        old_ll = 0
        for iter in range(max_iter):

            # E-step
            log_p = np.array([dnorm(data, mu=self.mus[cluster], sigma=self.sigmas[cluster]) + np.log(self.pi[cluster])
                          for cluster in range(self.n_clusters)])
            max_p = np.max(log_p, axis=0)
            sum_p = np.log(np.sum(np.exp(log_p - max_p), axis=0)) + max_p
            log_p -= sum_p
            p = np.exp(log_p)
            # Sample z
            z = np.zeros(n_data)
            for j in range(n_data):
                prob = np.around(p[:, j], 8)
                z[j] = np.argwhere(ss.multinomial.rvs(p=prob / np.sum(prob), n=1, size=1).reshape(self.n_clusters) == 1).item()

            # M-step: Update parameters by sampled z
            self.pi = np.array([np.mean(z == cluster) for cluster in range(self.n_clusters)])
            self.mus = [np.mean(data[z == cluster, :], axis=0) for cluster in range(self.n_clusters)]
            self.sigmas = [np.cov(data[z == cluster, :].T) for cluster in range(self.n_clusters)]

            # Calculate (negative) log_likelihood
            new_ll = -np.sum(sum_p)
            if abs(new_ll-old_ll) <= threshold:
                break
            else:
                old_ll = new_ll

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

    def _initialization(self, data, max_iter=10):
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




