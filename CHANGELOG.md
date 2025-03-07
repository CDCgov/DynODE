# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Calendar Versioning](https://calver.org/) using the following
pattern: `YYYY.MM.DD.micro`, where `micro` monotonically increases with each PR opened
on a given day. The `micro` version is suffixed with an `a` in the case of merging to `main` or
`development` branches, a `b` when starting the release process in the staging branch, and
no suffix when releases and the staging branch is pulled into the release branch.

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
