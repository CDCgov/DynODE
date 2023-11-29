# CFA Scenarios Model
[Overview](#overview) |
[Model Structure](#model-structure) |
[Quick Start](#quick-start) |
[Data Sources](#data-sources) |
[Project Admins](#project-admins) |
[Fine Text and Disclaimers](#general-disclaimer)
## Overview

This repository is for the design and implementation of a Scenarios forecasting model, built by the Scenarios team within CFA-Predict.

This code aims to combine a number of different codebases to forecast different covid scenarios with a Compartmental Mechanistic ODE model modeling multiple competing covid variants. The aim of this model is to provide enough flexibility for its users to explore a variety of scenarios, but also making certain design decisions that allow for fast computation and fitting as well as code readability.

What this model is:

a compartmental mechanistic ODE model capible of dynamic age binning, waning, vaccination scenarios, introduction of new variants, transmission structures, and timing estimation. TODO

What this model is not:

A fully dynamic suite of compartment models where any compartment may be easily added or removed. All models have assumptions, the basic compartment structure is assumed in many places, making it non-trivial to change.

## Model Structure

Subject to change, current transmission dynamics follow this basic model
![](/misc/scenarios_seip_model_diagram_cdc_blue.png)

## Quick Start

In order to run this model and get basic results follow these steps:
1. Import your config file, model ode, and `mechanistic_compartments.py`
2. Build a BasicMechanisticModel class with the builder or from scratch.
3. use the `.run` command to run without inference, and `.infer()` to use MCMC to fit some parameter values.

Here is an example script of a basic run of 100 days without inference of parameters, saving the simulation as an image to output/example.png:
```
from model_odes.seir_model_v5 import seirw_ode
from mechanistic_compartments import build_basic_mechanistic_model
from config.config_base import ConfigBase

solution = build_basic_mechanistic_model(ConfigBase()).run(seirw_ode, tf=100.0, show=True, save=True, save_path="output/example.png")
```

To create your own scenario, and modify parameters such as strain R0 and vaccination rate follow these steps:
1. Create a copy of config/config_scenario_template.py and name it whatever you would like.
2. Change `self.SCENARIO_NAME` to describe your new scenario
3. Modify or add any parameters you wish to differ from config/config_base.py
    1. if you wish to add parameters rather than just change them, you will need to pass that parameter to your model, through the `mechanistic_compartments.get_args()` function
4. Pass those parameter changes to the base constructor so it may initalize the rest to default values.
5. Run almost the same script as above, replacing your ConfigBase import with ConfigScenario.

```
from model_odes.seir_model_v5 import seirw_ode
from mechanistic_compartments import build_basic_mechanistic_model
from config.config_scenario_example import ConfigScenario

solution = build_basic_mechanistic_model(ConfigScenario()).run(seirw_ode, tf=100.0, show=True, save=True, save_path="output/example_scenario.png")
```

## Data Sources

The model is fed the following data sources:
1. data/demographic-data/contact_matricies : Dinas contact matricies todo
2. data/serological-data/* : serology data sourced from: [data.cdc.gov](https://data.cdc.gov/Laboratory-Surveillance/Nationwide-Commercial-Laboratory-Seroprevalence-Su/d2tw-32xv)

## Project Admins

Ariel Shurygin, M.S Data Sci, uva5@cdc.gov, CDC/IOD/ORR/CFA

Thomas Hladish, PhD, utx5@cdc.gov, CDC/IOD/ORR/CFA

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
