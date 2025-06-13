import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
from diffrax import Solution
from numpyro.infer import Predictive
from numpyro.infer.svi import SVIRunResult
from sir_age_stratified import SIRConfig, run_simulation

from dynode import MCMCProcess, Strain, SVIProcess
from dynode.config import SimulationConfig


def model(
    config: SimulationConfig,
    tf,
    obs_data: jax.Array | None = None,
):
    """Numpyro model for simulating infection incidence of an SIR model."""
    solution: Solution = run_simulation(config, tf)
    # compare to observed data if we have it
    assert solution.ys is not None, "mypy assert"
    incidence = jnp.diff(
        solution.ys[config.idx.r], axis=0
    )  # leading time axis
    incidence = jnp.maximum(incidence, 1e-6)
    numpyro.sample(
        "inf_incidence",
        numpyro.distributions.Poisson(incidence),
        obs=obs_data,
    )
    return solution


class SIRInferedConfig(SIRConfig):
    """An SIR config class with priors on r0 and infectious_period."""

    def __init__(self):
        """Set parameters for a infered SIR compartmental model.

        This includes compartment shape, initializer, and solver/transmission parameters.
        """
        # build the static version then replace the strain with
        # one modeled by some proposed priors instead.
        super().__init__()

        self.parameters.transmission_params.strains = [
            Strain(
                strain_name="swo9",
                r0=dist.TransformedDistribution(
                    dist.Beta(0.5, 0.5),
                    dist.transforms.AffineTransform(1.5, 1),
                ),
                infectious_period=dist.TruncatedNormal(
                    loc=8, scale=2, low=2, high=15
                ),
            )
        ]


if __name__ == "__main__":
    # produce synthetic data with fixed r0 and infectious period
    config_static = SIRConfig()
    solution = run_simulation(config_static, tf=100)
    # plot the soliution
    assert solution.ys is not None
    idx = config_static.idx
    # add 1 to each axis to account for the leading time dimension in `solution`
    plt.plot(
        jnp.sum(solution.ys[idx.s], axis=idx.s.age + 1),
        label="s",
    )
    plt.plot(
        jnp.sum(solution.ys[idx.i], axis=idx.i.age + 1),
        label="i",
    )
    plt.plot(
        jnp.sum(solution.ys[idx.r], axis=idx.r.age + 1),
        label="r",
    )
    plt.legend()
    plt.show()

    # diff recovered individuals to recover lagged incidence for each age group
    incidence = jnp.diff(solution.ys[idx.r], axis=0)
    # %%
    # set up inference process
    # now lets infer the parameters of this strain instead
    config_infer = SIRInferedConfig()
    # creating two InferenceProcesses, one for MCMC and one for SVI
    inference_process_mcmc = MCMCProcess(
        numpyro_model=model,
        num_warmup=500,  # higher = more accurate to a point
        num_samples=100,  # for posterior generation
        num_chains=1,
        nuts_max_tree_depth=10,
    )
    inference_process_svi = SVIProcess(
        numpyro_model=model,
        num_iterations=500,  # higher = more accurate to a point
        num_samples=100,  # for posterior generation
    )
    # %%
    # running inference
    print("fitting MCMC")
    inferer_mcmc = inference_process_mcmc.infer(
        config=config_infer, tf=100, obs_data=incidence
    )
    posterior_samples_mcmc = inference_process_mcmc.get_samples()
    # %%
    print("fitting SVI")
    inferer_svi = inference_process_svi.infer(
        config=config_infer, tf=100, obs_data=incidence
    )
    posterior_samples_svi = inference_process_svi.get_samples()

    # %%
    # printing results of inference
    print(
        f"Parameterized value of R0: {config_static.parameters.transmission_params.strains[0].r0} "
        f"Infectious Period: {config_static.parameters.transmission_params.strains[0].infectious_period}"
    )
    # notice the name of the posterior sample mimics the index of `transmission_params.strains`
    # this will help you find parameters later on.
    print(
        f"MCMC posterior's R0: {jnp.mean(posterior_samples_mcmc['strains_0_r0'])}, "
        f"Infectious Period: {jnp.mean(posterior_samples_mcmc['strains_0_infectious_period'])}"
    )
    print(
        f"SVI posterior's R0: {jnp.mean(posterior_samples_svi['strains_0_r0'])}, "
        f"Infectious Period: {jnp.mean(posterior_samples_svi['strains_0_infectious_period'])}"
    )
    svi_arviz = inference_process_svi.to_arviz()
    print(
        "the following arviz object is only interactive if run as a notebook."
    )
    print(svi_arviz)

    mcmc_arviz = inference_process_mcmc.to_arviz()
    axes = az.plot_density(
        [mcmc_arviz],
        data_labels=["R0"],
        var_names=["strains_0_r0"],
        shade=0.2,
    )

    fig = axes.flatten()[0].get_figure()
    fig.suptitle("Density Interval for R0")

    plt.show()
    print(
        "the following arviz object is only interactive if run as a notebook."
    )
    print(mcmc_arviz)
    # %%
    # projecting forward
    # now lets turn on Predictive mode and do some projections forward without observed data
    predictive_mcmc = Predictive(
        model,
        posterior_samples=posterior_samples_mcmc,
        exclude_deterministic=False,
    )
    posterior_incidence_mcmc = predictive_mcmc(
        rng_key=inference_process_mcmc.inference_prngkey,
        config=config_infer,  # arguments passed to `model`
        tf=200,
        obs_data=None,
    )
    assert inference_process_svi._inferer is not None, "mypy assert"
    assert isinstance(inference_process_svi._inference_state, SVIRunResult)
    predictive_svi = Predictive(
        model,
        guide=inference_process_svi._inferer.guide,
        params=inference_process_svi._inference_state.params,
        num_samples=1000,
    )
    posterior_incidence_svi = predictive_svi(
        rng_key=inference_process_mcmc.inference_prngkey,
        config=config_infer,  # arguments passed to `model`
        tf=200,
        obs_data=None,
    )

    # pick a random subset of 50 samples and plot the incidence, plot the true incidence from earlier as well
    random_samples = jax.random.choice(
        inference_process_mcmc.inference_prngkey,
        posterior_incidence_mcmc["inf_incidence"].shape[0],
        shape=(50,),
    )
    for sample in random_samples:
        plt.plot(
            jnp.sum(posterior_incidence_mcmc["inf_incidence"][sample], axis=1),
            label=None,
        )
    plt.plot(jnp.sum(incidence, axis=1), label="true incidence")
    plt.legend()
    plt.title("MCMC posterior predictive")
    plt.show()

    for sample in random_samples:
        plt.plot(
            jnp.sum(posterior_incidence_svi["inf_incidence"][sample], axis=1),
            label=None,
        )
    plt.plot(jnp.sum(incidence, axis=1), label="true incidence")
    plt.legend()
    plt.title("SVI posterior predictive")
    plt.show()
    # %%
