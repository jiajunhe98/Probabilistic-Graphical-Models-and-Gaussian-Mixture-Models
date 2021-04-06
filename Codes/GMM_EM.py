import numpy as np
import scipy.stats as ss


def dnorm(x, mu, sigma):
    sigma += np.eye(sigma.shape[0]) * 1e-8
    return ss.multivariate_normal.logpdf(x, mu, sigma)


class GMM_EM:
    """
    GMM by EM.

    Methods:
        fit(data, max_iter, threshold): Fit the model to data.
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

    def fit(self, data, max_iter=200, threshold=1e-8):
        """
        Fit the model to data.

        Args:
            data: Array-like, shape (n_samples, n_dim)
            max_iter: maximum number of EM steps.
            threshold: threshold to step iteration.
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

            # M-step
            self.pi = np.sum(p, axis=1) / n_data
            for cluster in range(self.n_clusters):
                effective_size = np.sum(p, axis=1)[cluster]
                self.mus[cluster] = np.sum(p[cluster].reshape(-1, 1) * data, axis=0) / effective_size
                self.sigmas[cluster] = ((data - self.mus[cluster]).T * p[cluster]) @ (data - self.mus[cluster]) / effective_size

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

    def _initialization(self, data, max_iter=50):
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




