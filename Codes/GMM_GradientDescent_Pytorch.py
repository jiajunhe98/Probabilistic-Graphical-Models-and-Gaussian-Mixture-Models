import numpy as np
import torch


def dnorm(x, mu, sigma):

    log_prob = -1 / 2 * (x - mu).T @ torch.inverse(sigma) @ (x - mu) \
               - 1 / 2 * torch.log(torch.det(sigma)) \
               - sigma.shape[0] / 2 * torch.log(torch.tensor(2 * np.pi))
    return log_prob.item()

def dGMM(x, pi, mus, sigmas):

    log_probs = torch.zeros(len(mus))
    for i in range(len(mus)):
        log_probs[i] += dnorm(x, mus[i], sigmas[i]) + torch.log(pi)[i]
    max_log = torch.max(log_probs)
    log_probs = torch.exp(log_probs - max_log)
    log_prob = torch.sum(log_probs) + max_log

    return log_prob


class GMM_GD:
    """
    GMM by Gradient Descent.

    Methods:
        fit(data, learning_rate, max_iter): Fit the model to data.
        predict(x): Predict cluster labels for x.
    """

    def __init__(self, n_clusters, device="cpu"):
        """
        Constructor Methods:

        Args:
            n_clusters(int): number of clusters.
            device: device used for calculating tensor by pytorch.
        """

        self.n_clusters = n_clusters
        self.device = device

        self.pi = None
        self.mus = [None] * self.n_clusters
        self.sigmas = [None] * self.n_clusters

    def fit(self, data, learning_rate=0.01, max_iter=100):
        """
        Fit the model to data.

        Args:
            data: Array-like data, shape (n_samples, n_dim)
            learning_rate: learning rate for Gradient Descent
            max_iter: max epoch for Gradient Descent

        """

        # Initialization by K-means
        z = self._initialization(data)
        self.pi = torch.tensor([np.mean(z == cluster)
                                for cluster in range(self.n_clusters)], requires_grad=True).to(self.device)
        self.mus = [torch.tensor(np.mean(data[z == cluster, :], axis=0).reshape(-1, 1), requires_grad=True).to(self.device)
                    for cluster in range(self.n_clusters)]
        self.sigmas = [torch.tensor(np.cov(data[z == cluster, :].T), requires_grad=True).to(self.device)
                       for cluster in range(self.n_clusters)]

        # Gradient Descent by Pytorch
        optimizer = torch.optim.SGD(params=[self.pi] + self.mus + self.sigmas, lr=learning_rate)
        lambd_pi = 1
        lambd_sigmas = [1] * self.n_clusters
        for iter in range(max_iter):
            optimizer.zero_grad()
            old_pi = self.pi
            log_likelihood = 0
            for x in data:
                x = torch.from_numpy(x.reshape(-1, 1)).to(self.device)
                log_likelihood += dGMM(x, self.pi, self.mus, self.sigmas)
            L = -log_likelihood + lambd_pi * torch.linalg.norm(self.pi)
            for cluster in range(self.n_clusters):
                L -= lambd_sigmas[cluster] * torch.det(self.sigmas[cluster])

            lambd_pi += learning_rate * torch.linalg.norm(self.pi)
            for cluster in range(self.n_clusters):
                lambd_sigmas[cluster] = max(lambd_sigmas[cluster] + learning_rate * torch.det(self.sigmas[cluster]), 0)
            L.backward()
            optimizer.step()
            if torch.linalg.norm(old_pi - self.pi) <= 1e-5:
                break
        return self

    def predict(self, x):
        """
        Predict cluster labels for x.

        Args:
            x: Array-like, shape (n_samples, n_dim)
        Return:
            Array-like, shape (n_samples, )
        """

        log_prob = [[dnorm(torch.from_numpy(d.reshape(-1, 1)).to(self.device), self.mus[cluster].detach(),
                           self.sigmas[cluster].detach()) + np.log(self.pi.detach())[cluster] for d in x]
                     for cluster in range(self.n_clusters)]
        log_prob = np.array(log_prob)
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
