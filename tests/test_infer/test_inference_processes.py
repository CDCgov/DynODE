import numpy as np
import numpyro
import numpyro.distributions as dist

import dynode.infer as infer

y = np.random.randn(128)  # from standard normal


def model(y):
    dist_loc = numpyro.sample("dist_loc", dist.Normal(1, 1))
    numpyro.sample("obs", dist.Normal(dist_loc, 1), obs=y)


def test_mcmc_process():
    mcmc_process = infer.MCMCProcess(
        numpyro_model=model,
        num_samples=10,
        num_chains=1,
        num_warmup=10,
        progress_bar=False,
        nuts_max_tree_depth=10,
    )

    mcmc_process = mcmc_process.infer(y=y)
    # completes, no raises.


def test_mcmc_process_get_samples():
    mcmc_process = infer.MCMCProcess(
        numpyro_model=model,
        num_samples=100,
        num_chains=1,
        num_warmup=50,
        progress_bar=False,
        nuts_max_tree_depth=10,
    )

    mcmc_process = mcmc_process.infer(y=y)
    samples = mcmc_process.get_samples()
    assert "dist_loc" in samples.keys()
    assert len(samples["dist_loc"]) == mcmc_process.num_samples


def test_svi_process():
    svi_process = infer.SVIProcess(
        numpyro_model=model,
        num_iterations=10,
        num_samples=10,
        progress_bar=False,
    )
    svi_process = svi_process.infer(y=y)


def test_svi_process_get_samples():
    svi_process = infer.SVIProcess(
        numpyro_model=model,
        num_iterations=10,
        num_samples=10,
        progress_bar=False,
    )
    svi_process.infer(y=y)
    samples = svi_process.get_samples()
    assert "dist_loc" in samples.keys()
    assert len(samples["dist_loc"]) == svi_process.num_samples
