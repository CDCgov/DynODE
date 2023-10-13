# cfa-scenarios-model
 
This repository is for the design and implementation of a Scenarios forecasting model, build by the Scenarios team within CFA-Predict.

This code aims to combine a number of different codebases to forecast different covid scenarios in a Compartmental Mechanistic ODE model with multiple competing covid variants.

# Quick Start

In order to run this model and get basic results follow these steps:
1. Import your config file, model ode, and `mechanistic_compartments.py`
2. Build a BasicMechanisticModel class with the builder or from scratch.
3. use the `.run` command to run without inference, and `.infer()` to use MCMC to fit some parameter values.

Here is an example script of a basic run without inference of parameters, saving the simulation as an image to output/example.png:
```
from model_odes.seir_model_v4 import seirw_ode
from mechanistic_compartments import build_basic_mechanistic_model
from config.config_base import ModelConfig

solution = build_basic_mechanistic_model(ModelConfig).run(seirw_ode, save_path="output/example.png")
```

# Data Sources

The model is fed the following data sources:
1. data/demographic-data/contact_matricies : Dinas contact matricies todo
2. data source 2 : used in the following way todo


# Model Structure

Subject to change, currently seir_model_v4 follows these disease dynamics.
![](/misc/seir_model_v4_diagram.png)

# Contact Developers

Core developers on this repository are:

1. Ariel (Arik) Shurygin
2. Thomas Hladish
