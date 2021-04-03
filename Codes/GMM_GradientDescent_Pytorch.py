import numpy as np
import torch

def dnorm(x, mu, sigma):
    log_prob = - 1 / 2 * torch.diag((x - mu).T @ torch.inverse(sigma) @ (x - mu)) \
               - 1 / 2 * torch.log(torch.det(sigma)) \
               - sigma.shape[0] / 2 * torch.log(torch.tensor(2 * np.pi))
    return log_prob


def dGMM(x, pi, mus, sigmas):

    log_probs = []
    for i in range(len(mus)):
        log_probs.append(dnorm(x, mus[i], sigmas[i]) + torch.log(pi)[i])
    log_probs = torch.vstack(log_probs)
    max_log = torch.max(log_probs, dim=0, keepdim=True)[0]
    probs = torch.exp(log_probs - max_log)
    log_prob = torch.log(torch.sum(probs, dim=0, keepdim=True)) + max_log

    return log_prob.reshape(-1)


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

    def fit(self, data, learning_rate=0.001, max_iter=5000, return_losses=True):
        """
        Fit the model to data.

        Args:
            data: Array-like data, shape (n_samples, n_dim)
            learning_rate: learning rate for Gradient Descent
            max_iter: max epoch for Gradient Descent

        """

        # Initialization pi and means by K-means
        z = self._initialization(data)
        self._pi_ = torch.tensor([np.log(np.mean(z == cluster))
                                for cluster in range(self.n_clusters)], requires_grad=True).to(self.device)
        self.pi = torch.exp(self._pi_) / torch.sum(torch.exp(self._pi_))
        self.mus = [torch.tensor(np.mean(data[z == cluster, :], axis=0).reshape(-1, 1), requires_grad=True).to(self.device)
                    for cluster in range(self.n_clusters)]

        # Initialization sigmas as unit matrix
        self._sigmas_ = [torch.tensor(np.eye(data.shape[1]), requires_grad=True).to(self.device)
                         for cluster in range(self.n_clusters)]
        self.sigmas = [self._sigmas_[cluster] @ self._sigmas_[cluster].T
                       for cluster in range(self.n_clusters)]

        # Gradient Descent by Pytorch
        optimizer = torch.optim.SGD(params=[self._pi_] + self.mus + self._sigmas_, lr=learning_rate)
        L = []
        for iter in range(max_iter):
            x = torch.from_numpy(data.T).to(self.device)
            minus_log_likelihood = torch.sum(-dGMM(x, self.pi, self.mus, self.sigmas))
            minus_log_likelihood.backward()
            L.append(minus_log_likelihood.item())
            optimizer.step()

            grad_norm = np.linalg.norm(self._pi_.grad.detach()) \
                        + sum([np.linalg.norm(self.mus[cluster].grad.detach())
                               + np.linalg.norm(self._sigmas_[cluster].grad.detach())
                               for cluster in range(self.n_clusters)])

            optimizer.zero_grad()

            self.pi = torch.exp(self._pi_) / torch.sum(torch.exp(self._pi_))
            self.sigmas = [self._sigmas_[cluster] @ self._sigmas_[cluster].T
                           for cluster in range(self.n_clusters)]

            if grad_norm <= 1e-5:
                break
        if return_losses:
            return L, self
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