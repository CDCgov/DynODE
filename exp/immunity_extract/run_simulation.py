from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

from config.config import Config
from mechanistic_model.mechanistic_inferer import MechanisticInferer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from model_odes.seip_model import seip_ode2_immunity_extract

jax.config.update("jax_enable_x64", True)
plt.switch_backend("agg")


class ImmunityExtractor(MechanisticInferer):
    def __init__(
        self,
        global_variables_path: str,
        inferer_path: str,
        runner: MechanisticRunner,
    ):
        """A specialized init method which does not take an initial state, this is because
        posterior particles will contain the initial state used."""
        inferer_json = open(inferer_path, "r").read()
        global_json = open(global_variables_path, "r").read()
        self.config = Config(global_json).add_file(inferer_json)
        self.runner = runner

    @partial(jax.jit, static_argnums=(0))
    def vaccination_rate(self, t):
        return jnp.zeros(
            (self.config.NUM_AGE_GROUPS, self.config.MAX_VACCINATION_COUNT + 1)
        )


def create_initial_state(config):
    dim_age = config.NUM_AGE_GROUPS
    dim_hist = config.NUM_STRAINS + 1
    dim_vax = config.MAX_VACCINATION_COUNT + 1
    dim_strain = config.NUM_STRAINS
    dim_wane = config.NUM_WANING_COMPARTMENTS
    s = jnp.zeros((dim_age, dim_hist, dim_vax, dim_wane))
    e = jnp.zeros((dim_age, dim_hist, dim_vax, dim_strain))
    i = jnp.zeros((dim_age, dim_hist, dim_vax, dim_strain))
    c = jnp.zeros((dim_age, dim_hist, dim_vax, dim_wane, dim_strain))
    s = s.at[:, :, :, 0].add(1)
    i = i.at[:, 0, 0, :].add(1)

    return (s, e, i, c)


if __name__ == "__main__":
    GLOBAL_CONFIG_PATH = "exp/immunity_extract/config_global.json"
    INFERER_CONFIG_PATH = "exp/immunity_extract/config_inferer.json"
    runner = MechanisticRunner(seip_ode2_immunity_extract)
    inferer = ImmunityExtractor(
        GLOBAL_CONFIG_PATH, INFERER_CONFIG_PATH, runner
    )
    inferer.config.CONTACT_MATRIX = jnp.array([[1.0]])
    inferer.config.POPULATION = jnp.ones((1,))
    inferer.config.INTRODUCTION_AGE_MASK = jnp.array(
        [
            True,
        ]
    )

    initial_state = create_initial_state(inferer.config)
    inferer.INITIAL_STATE = initial_state
    parameters = inferer.get_parameters()

    solution = runner.run(initial_state, tf=800, args=parameters)
    foi = jnp.sum(jnp.diff(solution.ys[3], axis=0), -2)
    protection = 1 - foi

    pdf_filename = "./output/immunity_profile.pdf"
    colors_vac = ["#e41a1c", "#377eb8", "#4daf4a"]

    pdf_pages = PdfPages(pdf_filename)
    for strain in range(foi.shape[4]):
        fig, axs = plt.subplots(3, 2)
        for hist in range(foi.shape[2]):
            i = hist // 2
            j = hist % 2
            axs[i, j].set_prop_cycle(cycler(color=colors_vac))
            axs[i, j].set_ylim([0.0, 1.0])
            axs[i, j].set_ylabel("Protection")

            y = protection[:, 0, hist, :, strain]
            if hist == 0:
                title = "No previous infection"
                axs[i, j].plot(
                    range(len(foi)), y, label=["vac 0", "vac 1", "vac 2"]
                )
            else:
                title = f"Last infected by strain {hist-1}"
                axs[i, j].plot(range(len(foi)), y)

            axs[i, j].hlines(
                y=0.25,
                xmin=0,
                xmax=len(foi),
                color="#888888",
                linestyle="dashed",
                alpha=0.3,
            )
            axs[i, j].set_title(title)

        fig.suptitle(f"Challenge by strain {strain}")
        fig.legend()
        # fig.tight_layout()
        fig.set_size_inches(8, 10)
        fig.set_dpi(300)

        pdf_pages.savefig(fig)
        plt.close(fig)
    pdf_pages.close()
