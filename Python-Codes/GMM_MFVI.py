import numpy as np
import scipy.stats as ss


def dnorm(x, mu, sigma):
    sigma += np.eye(sigma.shape[0]) * 1e-8
    return ss.multivariate_normal.logpdf(x, mu, sigma).reshape(1, -1)


class GMM_MFVI:

    def __init__(self, n_cluster):

        self.n_cluster = n_cluster

        self.pi = None
        self.mus = [None] * self.n_cluster
        self.sigmas = [None] * self.n_cluster

    def fit(self, data, max_iter=200):

        assert data.ndim == 2
        n_sample = data.shape[0]
        n_dim = data.shape[1]

        # Set prior distribution
        alpha = np.full((self.n_cluster, ), 2)     # Dirichlet Prior
        mu0, lambd, scale, df = np.zeros(n_dim), 1, np.eye(n_dim), n_dim     # NIW Prior

        # Initialization z
        z = self.initialization(data)

        for iter in range(max_iter):
            # Update pi as the mode of Dirichlet distribution
            self.pi = alpha + np.array([np.sum(z == i) for i in range(self.n_cluster)]) - 1
            self.pi = self.pi / np.sum(self.pi)

            # Update mu, sigma as the mode of NIW distribution
            for cluster in range(self.n_cluster):
                samples = data[z == cluster, :]
                n = samples.shape[0]
                mean = np.mean(samples, axis=0, keepdims=True) if n != 0 else None

                # Calculate Posterior NIW Distribution's parameters
                mu0_post = (lambd * mu0 + n * mean) / (lambd + n) if n != 0 else mu0
                lambd_post = lambd + n
                df_post = df + n
                scale_post = scale + np.dot((samples - mean).T, (samples - mean))\
                             + lambd * n / lambd_post * np.dot((mean-mu0).T, (mean-mu0)) if n != 0 else scale

                # Update mu, sigma as the mode of NIW distribution
                self.mus[cluster] = mu0_post.reshape((3, ))
                self.sigmas[cluster] = scale_post / (n_dim + df_post + 1)

            # Update z
            z = self.predict(data)

        return self

    def predict(self, x):
        log_prob = [dnorm(x, self.mus[cluster], self.sigmas[cluster]) + np.log(self.pi[cluster])
                     for cluster in range(self.n_cluster)]
        log_prob = np.vstack(log_prob)
        z = np.argmax(log_prob, axis=0)
        return z

    def initialization(self, data, max_iter=100):

        means = data[np.random.choice(data.shape[0], self.n_cluster, replace=False)]    # pick random samples as center
        z = np.zeros(data.shape[0])

        for iter in range(max_iter):
            dist = [np.sum((data - means[cluster]) ** 2, axis=1) for cluster in range(self.n_cluster)]
            dist = np.vstack(dist)
            z = np.argmin(dist, axis=0)
            means = [np.mean(data[z == cluster], axis=0) for cluster in range(self.n_cluster)]

        return z


