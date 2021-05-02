import numpy as np
import torch
import copy

def dinvwishart(x, scale, df):
    log_prob = torch.logdet(scale) * (df / 2) - np.log(2) * (scale.shape[0] * df / 2) - torch.mvlgamma(torch.tensor(df / 2), p=scale.shape[0]) + torch.logdet(x) * -(df + scale.shape[0] + 1) / 2 - 0.5 * torch.trace(scale @ torch.inverse(x))
    return log_prob

def dnorm(x, mu, sigma):
    log_prob = torch.distributions.multivariate_normal.MultivariateNormal(mu, sigma).log_prob(x)
    return log_prob

def dGMM(x, pi, alpha, mus, sigmas, mu0, lambd, df, scale):


    log_probs = []
    for i in range(len(mus)):
        log_probs.append(dnorm(x, mus[i], sigmas[i]) + torch.log(pi)[i])
    log_probs = torch.vstack(log_probs)
    max_log = torch.max(log_probs, dim=0, keepdim=True)[0]
    probs = torch.exp(log_probs - max_log)
    log_prob = torch.log(torch.sum(probs, dim=0, keepdim=True)) + max_log
    log_prob = torch.sum(log_prob)

    log_prob += torch.distributions.Dirichlet(alpha).log_prob(pi)
    for i in range(len(mus)):
        log_prob += dinvwishart(sigmas[i], scale, df)
        log_prob += torch.distributions.multivariate_normal.MultivariateNormal(mu0, sigmas[i] / lambd).log_prob(mus[i])
    return log_prob

def softmax(x):
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp)


class GMM_HMC:
    """
    GMM by HMC.

    Methods:
        fit(data, n_samples, burning_time): Fit the model to data.
        predict(x): Predict cluster labels for x.
    """

    def __init__(self, n_clusters, device="cpu"):
        """
        Constructor Methods:

        Args:
            n_clusters(int): number of clusters.
            device
        """

        self.n_clusters = n_clusters
        self.device = device

        self.pi = None
        self.mus = [None] * self.n_clusters
        self.sigmas = [None] * self.n_clusters

    def fit(self, data, n_samples=100, burning_time=100, epsilon=0.01):
        """
        Fit the model to data.

        Args:
            data: Array-like, shape (n_samples, n_dim)
            n_samples(int): Amount of samples
            burning_time(int): Times of sampling in burning time(before mixing)
            epsilon: stride for frog-leap
        """

        assert data.ndim == 2
        n_dim = data.shape[1]
        n_data = data.shape[0]

        # Set prior distribution
        alpha = torch.full((self.n_clusters, ), 1.0)     # Dirichlet Prior
        mu0, lambd, scale, df = torch.zeros(n_dim), 1, torch.eye(n_dim), n_dim     # NIW Prior

        # Initialize samples
        z = self._initialization(data)
        pi_sample = torch.tensor(np.log(np.array([np.mean(z == cluster) for cluster in range(self.n_clusters)])), requires_grad=True, dtype=torch.float).to(self.device)
        mus_sample = [torch.tensor(np.mean(data[z == cluster, :], axis=0), requires_grad=True, dtype=torch.float).to(self.device) for cluster in range(self.n_clusters)]
        sigmas_sample = [torch.tensor(np.linalg.cholesky(np.cov(data[z == cluster, :].T)), requires_grad=True, dtype=torch.float).to(self.device) for cluster in range(self.n_clusters)]

        data = torch.from_numpy(data).float().to(self.device)

        # Only keep the sample that give the max posterior probability
        P_MAP = -np.inf
        self.pi, self.mus, self.sigmas = pi_sample.clone(), [i.clone() for i in mus_sample], [i.clone() for i in sigmas_sample]

        for sample_times in range(n_samples + burning_time):

            old_pi_sample = pi_sample.clone().detach().requires_grad_()
            old_mus_sample = [i.clone().detach().requires_grad_() for i in mus_sample]
            old_sigmas_sample = [i.clone().detach().requires_grad_() for i in sigmas_sample]

            # Sample momentum
            m_pi = torch.randn_like(pi_sample)
            m_mu = [torch.randn_like(mus_sample[cluster]) for cluster in range(self.n_clusters)]
            m_sigma = [torch.randn_like(sigmas_sample[cluster]) for cluster in range(self.n_clusters)]

            H0 = 0.5 * (torch.sum(m_pi ** 2)
                        + sum(torch.sum(m_mu[cluster] ** 2) for cluster in range(self.n_clusters))
                        + sum(torch.sum(m_sigma[cluster] ** 2) for cluster in range(self.n_clusters)))\
                 - dGMM(data, softmax(pi_sample), alpha, mus_sample, [L@L.T for L in sigmas_sample], mu0, lambd, df, scale)

            # Update by Leaf Frog
            for step in range(int(1/epsilon)):
                optimizer = torch.optim.SGD([pi_sample] + sigmas_sample + mus_sample, lr=0)
                optimizer.zero_grad()
                E = -dGMM(data, softmax(pi_sample), alpha, mus_sample, [L@L.T for L in sigmas_sample], mu0, lambd, df, scale)
                E.backward()

                m_pi -= epsilon * pi_sample.grad
                pi_sample.requires_grad = False
                pi_sample += epsilon * m_pi
                pi_sample.requires_grad = True
                for cluster in range(self.n_clusters):
                    m_mu[cluster] -= epsilon * mus_sample[cluster].grad
                    m_sigma[cluster] -= epsilon * sigmas_sample[cluster].grad
                    mus_sample[cluster].requires_grad = False
                    sigmas_sample[cluster].requires_grad = False
                    mus_sample[cluster] += epsilon * m_mu[cluster]
                    sigmas_sample[cluster] += epsilon * m_sigma[cluster]
                    mus_sample[cluster].requires_grad = True
                    sigmas_sample[cluster].requires_grad = True

            H1 = 0.5 * (torch.sum(m_pi ** 2)
                        + sum(torch.sum(m_mu[cluster] ** 2) for cluster in range(self.n_clusters))
                        + sum(torch.sum(m_sigma[cluster] ** 2) for cluster in range(self.n_clusters)))\
                 - dGMM(data, softmax(pi_sample), alpha, mus_sample, [L@L.T for L in sigmas_sample], mu0, lambd, df, scale)

            # Check the Acceptance rate
            A = min(1, torch.exp(H0 - H1).item())
            u = np.random.rand(1)
            if u > A:
                pi_sample = old_pi_sample
                mus_sample = old_mus_sample
                sigmas_sample = old_sigmas_sample
            else:
                # If mixed, store the sample
                if sample_times >= burning_time:
                    P = dGMM(data, softmax(pi_sample), alpha, mus_sample, [L@L.T for L in sigmas_sample], mu0, lambd, df, scale)
                    if P > P_MAP:
                        self.pi, self.mus, self.sigmas, P_MAP = pi_sample.clone(), [i.clone() for i in mus_sample], [i.clone() for i in sigmas_sample], P
        self.pi = softmax(self.pi).detach()
        for cluster in range(self.n_clusters):
            self.mus[cluster] = self.mus[cluster].detach()
            self.sigmas[cluster] = (self.sigmas[cluster] @ self.sigmas[cluster].T).detach()

        return self

    def predict(self, x):
        """
        Predict cluster labels for x.

        Args:
            x: Array-like, shape (n_samples, n_dim)
        Return:
            Array-like, shape (n_samples, )
        """
        log_prob = [dnorm(torch.from_numpy(x).float().to(self.device), self.mus[cluster], self.sigmas[cluster]) + np.log(self.pi[cluster])
                     for cluster in range(self.n_clusters)]
        log_prob = np.vstack(log_prob)
        z = np.argmax(log_prob, axis=0)
        return z

    def _initialization(self, data, max_iter=1):
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




