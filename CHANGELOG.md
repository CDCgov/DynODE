# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Calendar Versioning](https://calver.org/) using the following
pattern: `YYYY.MM.DD.micro`, where `micro` monotonically increases with each PR opened
on a given day. The `micro` version is suffixed with an `a` in the case of merging to `main` or
`development` branches, a `b` when starting the release process in the staging branch, and
no suffix when releases and the staging branch is pulled into the release branch.

## [2025.06.16.1a] - enforcing stricter `jax` versioning
### Changed
- jax version requirements to avoid odd state in which `diffrax = 0.7.0` while `jax=0.6.0` which causes unknown failures.

## [2025.06.11.3a] - adding more `dynode.utils` tests
### Added
- some tests for `dynode.utils.drop_keys_with_substring()`
### Changed
- removed some unused methods and code within `test_utils.py`

## [2025.06.11.2a] - adding unit tests for `dynode.simulation`
### Added
- some unit tests for `dynode.simulate()`

## [2025.06.11.1a] - unit test expansion.
### Added
- some unit tests for dynode.infer objects.

## [2025.06.09.3a] - more efficient `sir.py` example script.
### Changed
- modified our `examples/sir.py` script to work in jupyter notebooks, as a script, and in our github workflows, while minimizing its runtime.

## [2025.06.09.2a] - updating ruff linter to latest version
### Changed
- no behavior changes, just updating ruff linter and pre-commit git actions.

## [2025.06.09.1a] - unit testing `dynode.config` sub-module
### Added
- A boat load of unit tests for `dynode.config` submodule. This is not exhaustive and will likely be further expanded in the future

## [2025.06.03.2a] - disallowing duplicate compartment names in `SimulationConfig`
### Changed
- adding a validator to `SimulationConfig` that disallows compartments with the same names, as this breaks the `idx` enum and `get_compartment()` functions.

## [2025.06.03.1a] - adding validators to catch empty `Strains` list cases
### Changed
- Patching a hole in the `TransmissionParams` validator as well as in the `FullStratifiedImmuneHistoryDimension` and `LastStrainImmuneHistoryDimension` classes that allowed empty lists of strains to be passed.

## [2025.06.01.3a] - patching a validator within `TransmissionParams`
### Changed
- patching a hole in the `TransmissionParams` validator that allowed for poorly formatted `strain_interactions` dictionaries to make it through validation.

## [2025.06.01.2a] - exposing `seasonal_vaccination` bool flag in `VaccinationDimension`
### Changed
- `VaccinationDimension.seasonal_vaccination` will now return a bool flag whether or not the dimension tracks seasonal vaccination or not. This helpers with greater clarity to the user whether or not seasonal vaccination is enabled for that dimension or not.

## [2025.06.01.1a] - `name` field exposed in `Dimension` helper classes.
### Changed
- `VaccinationDimension`, `FullStratifiedImmuneHistoryDimension`, and `LastStrainImmuneHistoryDimension` now all expose the `name` field in their constructors, allowing users greater flexibility when naming their dimensions.

## [2025.05.29.1a] - `SubSaveAt` support for `odes.simulate()`
### Changed
- Changed `odes.simulate()` added `sub_save_indicies` and `save_step` optional parameters that are passed to `build_saveat` function.
- Added `odes.build_saveat()` to determine (`sub_save_indicies is not None`) if `SubSaveAt` object should be used when building the `SaveAt` object. `build_saveat()` can also optionally increment the time steps `SaveAt` saves states via the `save_step` argument.

## [2025.05.22.1a] - Dynode Module Reorg
### Meta Changes
- Realized that all our versions were tagged with 2024 up until now ! Versions should now have the correct year, whoops.

### Changed
- Reorganized `Dynode` into 5 major modules `config`,`infer`,`simulation`,`typing`, and `utils`.
    - Reorganized some files within `model_configuration` depending on functionality, split across `config` and `infer` modules.
- Removed and transfered all disease specific code out to a currently private repository named `DynODE-Models`, keeping `Dynode` as a repo for the framework only.
    - This includes many files within `src/dynode` such as `mechanistic_inferer.py` and `mechanistic_runner.py`
    - This includes much of the testing infrastructure built around the outdated classes, thus `tests/*` has either been moved or removed.
- Split up much of `src/dynode/utils.py` into separate files depending on functionality, some going to `DynODE-Models`
- Removed old framework example from `examples/` along with the example configs. Keeping `examples/sir.py` as an example of new framework.
    - This also allowed for the removal of the United States data held for the example script within `data/`.
- Removed all outdated files within `data_manipulation_scripts/`
- Exposed all methods and classes from within the modules to the top level `__all__`, meaning users, if they so wish, can import things out of Dynode without traversing the modules.
- Removed now-unnecessary imports to clean up dependency tree of this repo.

## [2025.05.02.1a] - Fixing `scale_initial_infections`
### Changed
- Realized our version number still said 2024, whoops!
- Updated SimulationDate, rather than behaving like an integer by overloading its `__sub__` and `__add__` methods. We now programatically scan all `SimulationConfig` objects for instances of `SimulationDate` and replace the instance with the integer representation. This prevents future bugs like the one with deepcopying objects or numpyro initialization strategies failing to work on distributions using `SimulationDate`.


## [2024.04.23.1a] - Fixing `scale_initial_infections`
### Changed
- Changed `abstract_parameters.scale_initial_infections()` to preserve age distributions before and after scaling the number of initial infections. Before we were scaling E and I compartments and then scaling S down by the gained infections in E and I, without realizing that S/E/I had different age distributions individually.

## [2024.04.17.1a] - Static args support in `chex.Dataclass`
### Changed
- Changed chex version to a custom fork that allows for static arguments within a chex dataclass. See branch [here](https://github.com/mishmish66/chex/tree/static_keynames) and PR describing functionality [here](https://github.com/google-deepmind/chex/pull/327). This is meant to be temporary until Chex merges the commit into main. They are however slow to update and I think this feature is important enough to justify merging the fork.

- Updated `SIR.py` to use the new `static_keynames` functionality for an example.

## [2024.04.11.1a] - Dynode Evolution Inference Processes
### Added
- Added the `InferenceProcess` class as well as two concrete instances of `InferenceProcess` named `MCMCProcess` and `SVIProcess` within `dynode.model_configuration.inference.py`. These classes provide functionality for fitting to observed data using MCMC or SVI. Both also can easily retrieve their posterior samples, and convert themselves to arviz `InferenceData` objects for visualization.
- Added `dynode.sample`, a module containing helper functions to recurssively search for and sample `numpyro.distributions.Distribution` objects, as well as search and resolve `dynode.typing.DeterministicParameter` objects. Both necessary steps for inference.
- Some more descriptive error text on `dynode.typing.SamplePlaceholderError`
- The `dynode.typing.ObservedData` type hint.
- Two simple SIR pre-packaged configs within `src.dynode.model_configration.pre_packaged`, one with static strain R0 and infectious_period, another with priors to infer.

### Changed
- Modified `examples/sir.py` to showcase simulation of synthetic data, followed by refitting to it. Displaying the various usecases of both `MCMCProcess` and `SVIProcess`.
- Moved sampling helper code out of `utils.py` and into `dynode.sample`.
- Allowed `dynode.model_configuration.Strain` objects to contain `ArrayLike` types, meaning when a strain samples its R0 value, the resulting value (which may be a jax tracer in certain contexts), will still be accepted by Pydantic.
- Moved the `InferenceProcess` class out of `dynode.model_configuration.config_definition` and into `dynode.model_configuration.inference`.
- Added arviz to `pyproject.toml` as a dependency.
---

## [2024.04.14.1a] - Added `idx` property to dynode objects.
### Added
- enum property to `SimulationConfig`, `Compartment`, and `Dimension` classes.
- enums are linked recursively, meaning you can chain calls like `config.enum.s.vax.v0`
  to get the index of the first vax bin within the s compartment's vax dimension.

### Changed
- `Compartment`, `Dimension`, and `Bin` classes are no longer allowed to have
names with spaces or begining with a number as this breaks enum functionality.
Also requiring names to be all alphanumeric or underscore.

## [2024.04.08.1a] - Adding `SimulationDate` object
### Added
- Added a new `SimulationDate` helper object to allow users to specify datetime-like
objects in place of integers when specifying behaviors like prior distributions over
date ranges (E.g. introduction date of a strain centered around some date).
- Added a new enviornment variable `DYNODE_INITIALIZATION_DATE({pid})` so all parts
of the program can read in the model's start date regardless of where they
are in the code.

---

## [2024.03.25.2a] - Adding `sir.py` example run
### Added
- added `examples/sir.py` to show new users how to create a basic SIR ODE compartmental model
and simulate it for some number of days. This will act as a building block for
future examples that get increasingly complex.

---
## [2024.03.013.1a] - Adding ODE support
### Added
- Added `dynode.ode` module containing the `ODEBase` class, allowing users to
subclass their ODEs, provide their d/dt function and pass ODEs to other
Dynode classes and functions for inference / solving.

- Added `AbstractODEParams` within `dynode.ode` to pass a `chex.dataclass`
to `diffrax.diffeqsolve()` method instead of a dictionary. Improving clarity
in what parameters are available from within the ODEs as well as reducing
memory usage from passing large dictionaries of mostly unused parameters.

- Added `dynode.model_odes.seip_ode` module containing an example of how
a user may implement odes by subclassing the `ODEBase` and `AbstractODEParams`
classes to define their own behavior, reusing the odes defined in
`dynode.model_odes.seip_model`.
---

## [2024.03.24.1a] - Adding more `CompartmentalConfig` validations
### Added
- Require that each `Compartment` object contains dimensions with unique names.
- Require that dimensions of `DiscretizedPositiveIntBin` be sorted and have no gaps in coverage.
- Adding a helper function to allow users to easily create dimensions of `WaneBin` named in the correct way.

### Changed
- Renaming some minor functions/parameters with better names.
---

## [2024.03.013.1a] - Adding ODE support
### Added
- Added `dynode.ode` module containing the `ODEBase` class, allowing users to
subclass their ODEs, provide their d/dt function and pass ODEs to other
Dynode classes and functions for inference / solving.

- Added `AbstractODEParams` within `dynode.ode` to pass a `chex.dataclass`
to `diffrax.diffeqsolve()` method instead of a dictionary. Improving clarity
in what parameters are available from within the ODEs as well as reducing
memory usage from passing large dictionaries of mostly unused parameters.

- Added `dynode.model_odes.seip_ode` module containing an example of how
a user may implement odes by subclassing the `ODEBase` and `AbstractODEParams`
classes to define their own behavior, reusing the odes defined in
`dynode.model_odes.seip_model`.
---

## [2024.03.07.2a] - Adding `Strain.introduction_ages_mask_vector` field
### Changed
- added `introduction_ages_mask_vector` field to the `Strain` class, providing an
internally useful binary mask for the age bins specified optionally within
`Strain.introduction_ages` field.

---

## [2024.03.07.1a] - Adding `transform` to DeterministicParameter
### Changed
- `DeterministicParameter` from `dynode.typing` now has a `transform` parameter
which allows users to define a transform function for their parameter.

---

## [2024.03.04.1a] - Dynode typing
### Changed
- moved `dynode.model_configuration.types` to `dynode.typing`
- moved types previously declared within `dynode.__init__` into `dynode.typing`
- formally logging the change of the type `PosteriorSample` to `PlaceholderSample`

---

## [2024.02.26.2a] - DeterministicParameter type
### Added
- A `DeterministicParameter` class within `dynode.model_configuration.types` meant to identify parameters whose values depend on a possibly not yet realized value from another parameter. E.g. a parameter that is equal to another parameter that is itself a sample from some prior distribution.

---

## [2024.02.26.1a] - PosteriorSample distribution
### Added
- A `types.py` module for declaring types to be used within DynODE config files.
- A `PosteriorSample` class within `dynode.model_configuration.types` meant to identify parameters whose values will be replaced by a Posterior sample from a previous fit of the model.

---

## [2024.02.25.1a] - Pydantic Config Groundwork
### Added
- The `src/dynode/model_configuration` module for creation of Python config classes to initialize the new DynODE framework.

### Changed
- Nothing yet, as the new DynODE framework is written it will reference the new config classes instead of old structure.

### Deprecated
- Old `config.py` file within `src/dynode`

### Removed

### Fixed

---

## [2024.02.06.0a] - DynODE Evolution Initialization
### Added
- Added this changelog
- Created the `development` branch for `DynODE` Evolution effort.

### Changed
- Updated CI and repo template to most recent CFA standards
- Constrained dependencies within `pyproject.toml` and untracked `*.lock` files

### Deprecated
- None

### Removed
- Removed `Bayeux` package and removed old methods using it.

### Fixed
- Fixed any bugs and `mypy` errors caused by upgrading package versioning

---

## [0.1.0] - pre-evolution DynODE
- Initial experimental work on the `DynODE`, implementing core functionalities in a rapidly iterating small scale form.
