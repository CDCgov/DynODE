# DynODE Inference Module

This document describes the core inference classes and helper utilities used in the DynODE framework for probabilistic compartmental modeling. These APIs are designed to facilitate model fitting, parameter sampling, and checkpointing of simulation states.

## Visual Representation

```{mermaid}
classDiagram
    %% Abstract base class
    class InferenceProcess {
        <<abstract>>
        +**numpyro_model**: Callable
        +inference_prngkey: Array
        +infer(**kwargs)
        +get_samples(group_by_chain=False, exclude_deterministic=True)
        +to_arviz()
        - _inference_complete: bool
        - _inferer: Optional[MCMC | SVI]
        - _inference_state: Optional[HMCState | SVIRunResult]
        - _inferer_kwargs: Optional[dict]
    }
    class MCMCProcess {
        +num_samples: int
        +num_warmup: int
        +num_chains: int
        +nuts_max_tree_depth: int
        +nuts_init_strategy: Callable
        +nuts_kwargs: dict
    }
    class SVIProcess {
        +num_iterations: int
        +num_samples: int
        +guide_class: Type[AutoContinuous]
        +guide_init_strategy: Callable
        +optimizer: _NumPyroOptim
        +progress_bar: bool
        +guide_kwargs: dict
    }

    %% Inheritance
    InferenceProcess --> MCMCProcess : subclass
    InferenceProcess --> SVIProcess : subclass

```

---

## Inference Classes (`dynode.infer.inference`)

### `InferenceProcess`

**Abstract base class** for all inference processes in DynODE.
Defines the interface for fitting a `numpyro_model` to data, retrieving posterior samples, and exporting results to [ArviZ](https://arviz-devs.github.io/arviz/) for diagnostics and visualization.

**Key Methods:**
- `infer(**kwargs)`: Abstract. Fit the model to data.
- `get_samples(group_by_chain=False, exclude_deterministic=True)`: Abstract. Retrieve posterior samples.
- `to_arviz()`: Abstract. Convert results to an `arviz.InferenceData` object.

---

### `MCMCProcess(InferenceProcess)`

Implements inference using Markov Chain Monte Carlo ([MCMC](https://num.pyro.ai/en/stable/mcmc.html)) with the [NUTS](https://num.pyro.ai/en/latest/mcmc.html#numpyro.infer.hmc.NUTS) sampler from NumPyro.

**Parameters:**
- `num_samples`, `num_warmup`, `num_chains`: Control MCMC sampling.
- `nuts_max_tree_depth`, `nuts_init_strategy`, `nuts_kwargs`: NUTS sampler configuration.
- `progress_bar`: Show progress during sampling.

**Key Methods:**
- `infer(**kwargs)`: Runs MCMC and stores the sampler state.
- `get_samples(group_by_chain=False, exclude_deterministic=True)`: Returns posterior samples, optionally grouped by chain and/or including deterministic sites.
- `to_arviz()`: Returns an `arviz.InferenceData` object with posterior, prior, and posterior predictive samples.

---

### `SVIProcess(InferenceProcess)`

Implements inference using Stochastic Variational Inference ([SVI](https://num.pyro.ai/en/stable/svi.html)) with NumPyro's autoguides.

**Parameters:**
- `num_iterations`, `num_samples`: Control SVI fitting and posterior sampling respectively.
- `guide_class`, `guide_init_strategy`, `guide_kwargs`: Guide configuration.
- `optimizer`: SVI optimizer (default: Adam).
- `progress_bar`: Show progress during fitting.

**Key Methods:**
- `infer(**kwargs)`: Runs SVI and stores the optimizer state.
- `get_samples(exclude_deterministic=True)`: Returns posterior samples from the variational guide. No chains are used in SVI, so `group_by_chain` is not applicable.
- `to_arviz()`: Returns an `arviz.InferenceData` object with prior, posterior predictive, and log-likelihood.


---

## Inference Gotchas and Tips
- For information on exactly what to put inside of `numpyro_model`, please refer to the library backend documentation, section on [NumPyro](#numpyro). As numpyro [sites](https://num.pyro.ai/en/stable/primitives.html#module-numpyro.primitives) are the primary mechanism for the solver/optimizer of each inference process to update and sample parameters.
- in the event that your sampler/optimzer

---

## Sampling and Resolution Utilities (`dynode.infer.sample`)

`sample_distributions(obj, rng_key=None, _prefix="")`

Recursively traverses a data structure, sampling any `numpyro.Distribution` objects found.
- Handles nested dicts, lists, and Pydantic models.
- Site names are constructed using the `_prefix` argument for traceability.

**Returns:**
A copy of `obj` with all distributions replaced by samples.

---

`resolve_deterministic(obj, root_params, _prefix="")`

Recursively resolves any `DeterministicParameter` objects in a data structure, replacing them with their computed values based on `root_params`.

**Returns:**
A copy of `obj` with all deterministic parameters resolved.

---

`sample_then_resolve(parameters, rng_key=None)`

Convenience function that:
1. Deep-copies `parameters` so that parallel chains of inference do not interfere with each other.
2. Samples all distributions
3. Resolves all deterministic parameters

**Returns:**
A fully concrete, JAX-compatible copy of `parameters`.

---

## Checkpointing Utilities (`dynode.infer.checkpointing`)

`checkpoint_compartment_sizes(config, solution, save_final_timesteps=True, compartment_save_dates=[])`

Records compartment sizes at specified simulation dates for debugging and analysis.

**Parameters:**
- `config`: The `SimulationConfig` used for the ODE simulation.
- `solution`: The `diffrax.Solution` object from ODE integration.
- `save_final_timesteps`: If `True`, saves the final value for each compartment.
- `compartment_save_dates`: List of `datetime.date` objects to checkpoint.

**Behavior:**
- Uses `numpyro.deterministic` to record compartment values at requested dates and/or at the final timestep.

---


## See Also
- [ArviZ documentation](https://arviz-devs.github.io/arviz/)
