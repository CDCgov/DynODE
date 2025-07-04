# DynODE : A Dynamic Ordinary Differential Package for Respiratory Disease Modeling

![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FCDCgov%2FDynODE%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&query=%24.tool.poetry.version&style=plastic&label=version&color=lightgray)
![pre-commit](https://github.com/CDCgov/dynode/workflows/pre-commit/badge.svg?style=plastic&link=https://github.com/CDCgov/dynode/actions/workflows/pre-commit.yaml)
[![pytest](https://github.com/CDCgov/DynODE/actions/workflows/pytest.yaml/badge.svg)](https://github.com/CDCgov/DynODE/actions/workflows/pytest.yaml)
![GitHub License](https://img.shields.io/github/license/cdcgov/dynode?style=plastic&link=https://github.com/CDCgov/dynode/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54&style=plastic)

[Overview](#overview) |
[Model Structure](#model-structure) |
[Quick Start](#quick-start) |
[Data Sources](#data-sources) |
[Project Admins](#project-admins) |
[Fine Text and Disclaimers](#general-disclaimer)

## Overview

> [!IMPORTANT]
> This repository/branch is under active development and is a part of a major refactoring of DynODE and surrounding repositories.
> Please look around, but we advise against working with this code until it has stabilized.

This repository is for the design and implementation of the DynODE framework for disease Scenario forecasting, built by the Scenarios team within CFA-Predict.

Currently, we aim to use this code to forecast different disease tranmission scenarios with a compartmental mechanistic ODE model. We aim to provide enough flexibility for the code users to explore a variety of scenarios, but also making certain design decisions that allow for fast computation and fitting as well as code readability.

This framework has already been used to create:

- a compartmental mechanistic ODE model that accounts for age structure, immunity history, vaccination, immunity waning and multiple variants.

What this framework is not:

A fully dynamic suite of compartment models that are easily interchangable or modified.

## Quick Start

To get a taste of what this framework is capable of start at `examples/example_sir_config.py` and see how you may define a basic SIR model with age structure (young and old).
`example_sir_config.py` contains the following:
- `SIRInitializer`: describes how the initial conditions of the model, in this case with hardcoded values, but often with real world data informing compartment initial conditions.
- `SIRConfig`: describes the compartment structure of the model, the 3 compartments, Susceptible, Infectious, and recovered, as well as their dimensions, young and old in this case.
- `SIRInferedConfig`: a copy of `SIRConfig` but with undefined strain transmissibility, meant to be infered based on observed data.

After understanding the compartments and initial values of the model, go to `examples/sir.py` to see a simple scenario in which we:
- Define some Ordinary Differential Equations (ODEs) that dictate movement between compartments.
- Simulate an infection timeline
- Noise the infections
- Fit back to the noised infections to retrieve our original R0 and infectious period.
- Display some posterior traceplots and other inference metrics.

## Technical Details

For a full in-depth description of the model please see the [Github Pages](https://github.com/cdcent/cfa-scenarios-model/wiki) of this repo, where a living document of the model is stored.

## Project Admins
Thomas Hladish, Lead Data Scientist, utx5@cdc.gov, CDC/IOD/ORR/CFA

Ariel Shurygin, Data Scientist, uva5@cdc.gov, CDC/IOD/ORR/CFA

Kok Ben Toh, Data Scientist, tjk3@cdc.gov, CDC/IOD/ORR/CFA

Michael Batista, Data Scientist, upi8@cdc.gov, CDC/IOD/ORR/CFA

## General Disclaimer
This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/organization/mission.htm).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

## Public Domain Standard Notice
This repository constitutes a work of the United States Government and is not
subject to domestic copyright protection under 17 USC § 105. This repository is in
the public domain within the United States, and copyright and related rights in
the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
All contributions to this repository will be released under the CC0 dedication. By
submitting a pull request you are agreeing to comply with this waiver of
copyright interest.

## License Standard Notice
This repository is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under
the terms of the Apache Software License version 2, or (at your option) any
later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this
program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

## Privacy Standard Notice
This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice
Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo)
and submitting a pull request. (If you are new to GitHub, you might start with a
[basic tutorial](https://help.github.com/articles/set-up-git).) By contributing
to this project, you grant a world-wide, royalty-free, perpetual, irrevocable,
non-exclusive, transferable license to all users under the terms of the
[Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or
later.

All comments, messages, pull requests, and other submissions received through
CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice
This repository is not a source of government records but is a copy to increase
collaboration and collaborative potential. All government records will be
published through the [CDC web site](http://www.cdc.gov).

## Additional Standard Notices
Please refer to [CDC's Template Repository](https://github.com/CDCgov/template)
for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/master/CONTRIBUTING.md),
[public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md),
and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
