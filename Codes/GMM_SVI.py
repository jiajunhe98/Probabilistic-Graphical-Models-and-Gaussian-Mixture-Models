import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
import pyro.optim as optim
import scipy.stats as ss
from pyro.infer.autoguide import AutoDelta
from pyro import poutine


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dnorm(x, mu, sigma):
    sigma += np.eye(sigma.shape[0]) * 1e-8
    return ss.multivariate_normal.logpdf(x, mu, sigma).reshape(1, -1)


class GMM_SVI:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.pi = None
        self.mus = None
        self.sigmas = None

    def fit(self, data, max_iter=300, lr=0.1):

        def init_loc_fn(site):
            if site["name"] == "pi":
                return torch.ones(self.n_clusters) / self.n_clusters
            if site["name"] == "sigma":
                return torch.eye(data.shape[1]).expand([self.n_clusters, data.shape[1], data.shape[1]])
            if site["name"] == "mu":
                return torch.tensor(self._initialization(data))
            raise ValueError(site["name"])

        data = torch.from_numpy(data).float()
        svi = SVI(model=self.model, guide=AutoDelta(
            poutine.block(self.model, expose=["pi", "mu", "sigma"]), init_loc_fn=init_loc_fn),
                  optim=optim.Adam({"lr": lr}),
                  loss=TraceEnum_ELBO())
        pyro.clear_param_store()
        for i in range(max_iter):
            svi.step(data)
        self.mus = [pyro.get_param_store()["AutoDelta.mu"].detach().numpy()[i] for i in range(self.n_clusters)]
        self.sigmas = [pyro.get_param_store()["AutoDelta.sigma"].detach().numpy()[i] @\
                       pyro.get_param_store()["AutoDelta.sigma"].detach().numpy()[i].T + np.eye(data.shape[1])*0.01
                       for i in range(self.n_clusters)]
        self.pi = pyro.get_param_store()["AutoDelta.pi"].detach().numpy()
        self.pi = self.pi / np.sum(self.pi)
        return self

    def predict(self, x):
        log_prob = [dnorm(x, self.mus[cluster], self.sigmas[cluster]) + np.log(self.pi[cluster])
                     for cluster in range(self.n_clusters)]
        log_prob = np.vstack(log_prob)
        z = np.argmax(log_prob, axis=0)
        return z

    @config_enumerate
    def model(self, data):
        pi = pyro.sample("pi", dist.Dirichlet(concentration=torch.ones(self.n_clusters)))
        with pyro.plate("cluster_loop", self.n_clusters):
            mu = pyro.sample("mu", dist.Normal(loc=torch.zeros(data.shape[1]), scale=10.).to_event(1))
            sigma = pyro.sample("sigma",
                                dist.Normal(loc=torch.eye(data.shape[1]), scale=10.).to_event(2))
        sigma = torch.bmm(sigma, sigma.permute(0, 2, 1))
        with pyro.plate("data_loop", len(data)):
            z = pyro.sample("z", dist.Categorical(probs=pi))
            obs = pyro.sample("obs", dist.MultivariateNormal(loc=mu[z], covariance_matrix=sigma[z]+torch.eye(data.shape[1])*0.01), obs=data)



    def _initialization(self, data, max_iter=2):
        """
        Initialization by K-Means.
        """
        data = data.numpy()
        means = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]    # pick random samples as center
        z = np.zeros(data.shape[0])

        for iter in range(max_iter):
            d = [np.sum((data - means[cluster]) ** 2, axis=1) for cluster in range(self.n_clusters)]
            d = np.vstack(d)
            z = np.argmin(d, axis=0)
            means = [np.mean(data[z == cluster], axis=0) for cluster in range(self.n_clusters)]

        mus = [np.mean(data[z == cluster], axis=0) for cluster in range(self.n_clusters)]

        return mus













