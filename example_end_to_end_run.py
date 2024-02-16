from mechanistic_model.covid_initializer import CovidInitializer
from mechanistic_model.mechanistic_runner import MechanisticRunner
from mechanistic_model.solution_iterpreter import SolutionInterpreter
from mechanistic_model.static_value_parameters import StaticValueParameters
from model_odes.seip_model import seip_ode

if __name__ == "__main__":
    GLOBAL_CONFIG_PATH = "config/config_global.json"
    INITIALIZER_CONFIG_PATH = "config/config_initializer_covid.json"
    RUNNER_CONFIG_PATH = "config/config_runner_covid.json"
    INTERPRETER_CONFIG_PATH = "config/config_interpreter_covid.json"
    # model = build_basic_mechanistic_model(ConfigBase())
    initializer = CovidInitializer(INITIALIZER_CONFIG_PATH, GLOBAL_CONFIG_PATH)
    static_params = StaticValueParameters(
        initializer.get_initial_state(),
        RUNNER_CONFIG_PATH,
        GLOBAL_CONFIG_PATH,
    ).get_parameters()
    runner = MechanisticRunner(seip_ode)
    solution = runner.run(
        initializer.get_initial_state(), tf=200, args=static_params
    )
    interpreter = SolutionInterpreter(
        solution, INTERPRETER_CONFIG_PATH, GLOBAL_CONFIG_PATH
    )
    fig, ax = interpreter.summarize_solution(
        plot_commands=["S[0, :, :, :]", "E[0, :, :, :]", "I[0, :, :, :]"]
    )
