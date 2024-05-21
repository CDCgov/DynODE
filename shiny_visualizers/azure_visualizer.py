"""
This Python script, when run, launches a Shiny app on http://localhost:8000/ that aids in visualizing
individual particles of posteriors produced by an experiment.
"""

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sn
# from shiny import App, render, ui
# import utils
import os

# from mechanistic_model.covid_initializer import CovidInitializer
from cfa_azure.clients import AzureClient

INPUT_BLOB_NAME = "scenarios-test-container"
OUTPUT_BLOB_NAME = "example-output-scenarios-mechanistic"
client = AzureClient(config_path="secrets/configuration_cfaazurebatchprd.toml")
client.set_input_container(INPUT_BLOB_NAME, "input")
client.set_output_container(OUTPUT_BLOB_NAME, "output")
container_client = client.out_cont_client
experiment_name = "fifty_state_2304_2404_3strain"
job_name = "smh_epoch_2_240517"
state_name = "AK"
path_for_blobs = os.path.join(experiment_name, job_name, state_name).replace("\\", "/")
print(path_for_blobs)
blobs = container_client.list_blobs(name_starts_with=path_for_blobs)
for blob in blobs:
    print(blob.name)

# app_ui = ui.page_fixed(
#     ui.h2("Visualizing Immune History"),
#     ui.markdown(
#         """
#     """
#     ),
#     ui.layout_sidebar(
#         ui.panel_sidebar(
#             ui.input_selectize(
#                 "compartment",
#                 "Compartment",
#                 compartment_choices,
#                 multiple=True,
#                 selected="Susceptible",
#             ),
#             ui.input_selectize(
#                 "age_bin",
#                 "Age Bins",
#                 age_choices,
#                 multiple=True,
#                 selected="All",
#             ),
#             ui.input_selectize(
#                 "display",
#                 "Values",
#                 ["Proportion", "Count"],
#                 selected="Proportion",
#             ),
#         ),
#         ui.panel_main(ui.output_plot("plot", height="750px")),
#     ),
# )


# def server(input, output, session):
#     @output
#     @render.plot
#     def plot():
#         fig, axs = plt.subplots(2, 1)
#         fig, axs[0] = heatmap(input, fig, axs[0])
#         fig, axs[1] = waning_in_population(input, fig, axs[1])
#         return fig


# app = App(app_ui, server)
# app.run()
