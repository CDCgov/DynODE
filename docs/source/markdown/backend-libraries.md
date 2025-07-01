# The Library Backend of DynODE

DynODE relies on several key libraries to provide its core functionality.

- [JAX](#jax)
- [Chex](#chex)
- [Numpyro](#numpyro)
- [Diffrax](#diffrax)

(jax)=
## JAX

from the [JAX documentation](https://jax.readthedocs.io/en/latest/)

"_JAX is a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning._"

### JAX in DynODE
JAX is the primary backend for DynODE, providing automatic differentiation and Just-in-Time (JIT) compilation capabilities. It allows for efficient computation on both CPU and GPU, making it suitable for large-scale simulations and inference tasks. The decision to use Numpyro and Diffrax is motivated by their interfacing with Jax. Thus this is the hardest dependency to remove from DynODE, as it is the primary backend for all computations.

One of the hardest concepts to grasp when working with JAX is the Just-In-Time (JIT) compilation. In short, at a certain point in execution, usually right before the inference loop begins, we will compile the `numpyro_model`, this can lead to a whole host of confusion that is not isolated to DynODE. Take a look at some FAQ's on the [JAX FAQ](https://jax.readthedocs.io/en/latest/faq.html) for some common gotchas on JIT compilation.

In short whenever you are working with values handed into your `numpyro_model` function, you can not concretize them. Things like basic if statements `if x > 0` or certain operations like `x.reshape(new_shape)` will not work. Now this does not mean that arithmetic operations like `x + 1` or `x * 2` will not work, because these operations are simply added to the computation graph and will be executed when the JIT compiled function is run.

### JAX Gotchas and Tips
- [DynODE functions must be pure](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)
- JAX Arrays look like numpy arrays, but are immutable, and thus [require a different schema for modification](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#in-place-updates)
- JAX HATES booleans, if you find yourself requiring one, look at [jnp.where](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.where.html) but be careful with NANs and how these calls can impact the Numpyro solver/optimizer.
- [Out of bounds indexing is allowed and weird](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#out-of-bounds-indexing)
- JAX arrays may be non-concrete, but their `shape` is always available. Meaning that you can still get `x.shape` if you need it.

### Additional Reading
If you are new to JAX, and want to literally read a book on it, I recommend [Deep Learning with JAX](https://ieeexplore.ieee.org/document/10745362) but for use cases like DynODE, the reading is not entirely necessary. The JAX documentation is quite good.

---
(chex)=
## Chex
From the [Chex documentation](https://chex.readthedocs.io/en/latest/) and [github](https://github.com/google-deepmind/chex)

_Dataclasses are a popular construct introduced by Python 3.7 to allow to easily specify typed data structures with minimal boilerplate code. They are not, however, compatible with JAX and [dm-tree](https://github.com/deepmind/tree) out of the box._

_In Chex we provide a JAX-friendly dataclass implementation reusing python [dataclasses](https://docs.python.org/3/library/dataclasses.html#module-dataclasses)._

_Chex implementation of dataclass registers dataclasses as internal [PyTree nodes](https://jax.readthedocs.io/en/latest/pytrees.html) to ensure compatibility with JAX data structures._

### Chex in DynODE

In DynODE, Chex is used as an intermediate layer between the user-supplied configuration and the JAX accelerated ODEs.

For example, a user defines a list of `Strain` objects, each with their own R0 and Infectious Period, however efficient ODEs require a vectorized form of these parameters so that linear algebra may accelerate the computation. The object holding all the vectorized parameters must also be a [Pytree](https://docs.jax.dev/en/latest/pytrees.html#what-is-a-pytree), to be accepted by our [simulate](https://github.com/CDCgov/DynODE/wiki/simulation#parameters) method.

Chex acts as a wrapper around the python `dataclass` and also serves as a `PyTree` for diffrax, how convenient!

### Chex Gotchas and Tips
- Unlike regular Python `dataclass` objects, chex dataclasses cannot accept objects it was not defined for, meaning if you want to pass a new vector to your ODEs, youll need to add it to your Chex parameter definition.
- By default Chex will require that whatever you pass into it is a JAX object, however just like JAX allows for static arguments to compiled functions, our fork of Chex also allows for the marking of certain ODE parameters as static. Hopefully this pull request will be merged into Chex proper eventually, but the library is quite slow to turn over so it is unlikely any time soon.

---
(numpyro)=
## Numpyro

from the [NumPyro documentation](https://num.pyro.ai/)

"_NumPyro is a lightweight probabilistic programming library that provides a NumPy backend for Pyro. We rely on JAX for automatic differentiation and JIT compilation to GPU / CPU. NumPyro is under active development, so beware of brittleness, bugs, and changes to the API as the design evolves._"

### Numpyro in DynODE
Numpyro is the primary inference backend for DynODE. Both SVI and MCMC are powered by NumPyro, and the `numpyro_model` is a user-supplied function that defines the probabilistic model. In DynODE, many transmission-related parameters are modeled initially as prior distributions, Numpyro handels sampling these distributions into concrete values. Sampled values are then used in the ODE solver to simulate the disease dynamics, and handed back to NumPyro to calculate the likelihood of the observed data given the simulated data. Lastly NumPyro will update the prior distributions based on the likelihood of the observed data, and repeat this process however many times the user has specified in their InferenceProcess.

Numpyro is accelerated by JAX, meaning all the previously described Just-in-Time (JIT) problems arise when working with these distribution samples. Once a prior distribution is sampled, the user can not concretize the value.

### Numpyro Gotchas and Tips

- When diagnosing inference issues [numpyro handelers](https://num.pyro.ai/en/stable/handlers.html) are your friend. Specificall [trace](https://num.pyro.ai/en/stable/handlers.html#trace) and [seed](https://num.pyro.ai/en/stable/handlers.html#seed). These effect handlers will allow you to execute the `numpyro_model` outside of the inference process, and intercept the observed likelihood before they go to the NUTS sampler or SVI optimzer.
- Numpyro's [sample](https://num.pyro.ai/en/stable/primitives.html#sample) method is the primary way to pull values from a prior distribution, be mindful the value you recieve will be a JAX array, and thus subject to all the JAX gotchas described above, mainly concretization and immutability.
- Often times you may want to freeze certain parameters, or set them to a specific value. This can be done by wrapping the `numpyro_model` with the [substitute effect handler](https://num.pyro.ai/en/stable/handlers.html#substitute). This allows you to replace a parameter with a fixed value, without modifying the original model or config. This is particular useful for sensitivity analysis, or generating infection timeseries from the posteriors after a fit.

---
(diffrax)=
## Diffrax

from the [diffrax documentation](https://docs.kidger.site/diffrax/):

_Diffrax is a JAX-based library providing numerical differential equation solvers._

_Features include:_
- _ODE/SDE/CDE (ordinary/stochastic/controlled) solvers;_
- _lots of different solvers (including Tsit5, Dopri8, symplectic solvers, implicit solvers);_
- _vmappable everything (including the region of integration);_
- _using a [PyTree](https://docs.jax.dev/en/latest/pytrees.html) as the state;_
- _dense solutions;_
- _multiple adjoint methods for backpropagation;_
- _support for neural differential equations._


DynODE uses diffrax to solve ODEs for compartmental models, while changing solvers is theoretically simple, nothing except for Tsit5 has been tested. In our case the PyTree state is a tuple of JAX arrays representing the compartment states, and the region of integration is the number of days to simulate._"

### Diffrax in DynODE
Diffrax is used by the `simulate` function in the `simulation` module to numerically solve the ODEs defined by the user. The `simulate` function returns a `diffrax.Solution` object containing the compartment states at each saved time point. The user must supply an ODE function that defines the RATE OF CHANGE of the compartments at a particular point, and Diffrax handles the integration over the specified duration.

### Diffrax Gotchas and Tips
- Diffrax handles integration over the duration, so do not return compartment sizes in your ODEs, return the rate of change in that compartment, meaning negative gradients are allowed as long as they dont lead to negative compartment sizes.
- Diffrax supports both constant and adaptive step size integration, which can be configured via the `SolverParams` object passed to the `simulate` function.
- While your ODEs may be a relatively small part of the total code written, they are run by far the most times per fit. If fitting for 1000 iterations, each fit being 1000 days, the adaptive step sizer will run many times more than just 1000 times per iteration. Thus, it is important to keep your ODEs as efficient as possible. Look into using as many JAX operations as possible, and avoid long python loops. Desiging ODEs for performance is where you need to put your software development hat on, and think about how to optimize your code for speed.
- It may happen that your step sizer fails to move, this will show up as the `simulate` function either hanging or erroring after hitting the maximum number of steps. Rather than immediately cranking the `max_steps` up, try to use `jax.debug.print()` to print out the compartment state and gradients for each time step, then look for negatives or NANs. Often you may have a misspecified parameter value, or a bug in your seasonality module or introductions that is causing crazy behavior in the ODEs.
