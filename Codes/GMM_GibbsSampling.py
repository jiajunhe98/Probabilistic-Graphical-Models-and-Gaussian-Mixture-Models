import numpy as np
import scipy.stats as ss


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


class GMM_GibbsSampling:
    """
    GMM by Gibbs Sampling.

    Methods:
        fit(data, max_iter): Fit the model to data.
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

    def fit(self, data, n_samples=200, burning_time=50):
        """
        Fit the model to data.

        Args:
            data: Array-like, shape (n_samples, n_dim)
            n_samples(int): Amount of samples by Gibbs Sampling
            burning_time(int): Times of sampling in burning time(before mixing)
        """

        assert data.ndim == 2
        n_dim = data.shape[1]
        n_data = data.shape[0]

        # Set prior distribution
        alpha = np.full((self.n_clusters, ), 2)     # Dirichlet Prior
        mu0, lambd, scale, df = np.zeros(n_dim), 1, np.eye(n_dim), n_dim     # NIW Prior

        # Initialize samples, only need to store samples of z
        z_sample = self._initialization(data)
        pi_sample = None
        mus_sample = [None] * self.n_clusters
        sigmas_sample = [None] * self.n_clusters
        z_samples = []

        for sample_times in range(n_samples + burning_time):
            # Update pi by sampling
            alpha_post = posterior_dirichlet(alpha, z_sample)
            pi_sample = ss.dirichlet.rvs(alpha=alpha_post, size=1).reshape((self.n_clusters, ))

            # Update mu, sigma by sampling
            for cluster in range(self.n_clusters):
                data_in_cluster = data[z_sample == cluster, :]

                # Calculate Posterior NIW Distribution's parameters
                mu0_post, lambd_post, df_post, scale_post = posterior_NIW(mu0, lambd, scale, df, data_in_cluster)

                # Update mu, sigma by sampling
                sigmas_sample[cluster] = ss.invwishart.rvs(df=df_post, scale=scale_post, size=1)
                mus_sample[cluster] = ss.multivariate_normal.rvs(
                    mean=mu0_post.reshape((n_dim, )), cov=sigmas_sample[cluster] / lambd_post, size=1)

            # Update z by sampling
            log_prob = [dnorm(data, mus_sample[cluster], sigmas_sample[cluster]) + np.log(pi_sample[cluster])
                        for cluster in range(self.n_clusters)]
            log_prob = np.vstack(log_prob)
            log_prob -= np.max(log_prob, axis=0, keepdims=True)
            prob = np.exp(log_prob) + 1e-8
            prob /= np.sum(prob, axis=0, keepdims=True)

            z_sample = np.array(
                [np.argwhere(ss.multinomial.rvs(p=prob[:, j], n=1, size=1).reshape(self.n_clusters) == 1).item()
                 for j in range(n_data)]
            )
            if sample_times >= burning_time:
                z_samples.append(z_sample)

        # Using samples to estimate z
        z = ss.mode(np.vstack(z_samples), axis=0)[0][0]

        # Use z to calculate pi, mu, sigma as the mode of the posterior distribution
        # Update pi as the mode of Dirichlet distribution
        alpha_post = posterior_dirichlet(alpha, z)
        self.pi = alpha_post - 1
        self.pi = self.pi / np.sum(self.pi)

        # Update mu, sigma as the mode of NIW distribution
        for cluster in range(self.n_clusters):
            data_in_cluster = data[z == cluster, :]

            # Calculate Posterior NIW Distribution's parameters
            mu0_post, lambd_post, df_post, scale_post = posterior_NIW(mu0, lambd, scale, df, data_in_cluster)

            # Update mu, sigma as the mode of NIW distribution
            self.mus[cluster] = mu0_post.reshape((n_dim,))
            self.sigmas[cluster] = scale_post / (n_dim + df_post + 1)

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




