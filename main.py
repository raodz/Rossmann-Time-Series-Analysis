from src.experiments import (get_experiment_definitions, execute_experiments,
                             prepare_environment)
from src.logging import (generate_summary, generate_model_comparison,
                         log_program_complete)
from src.utils import load_config


def main():
    prepare_environment()

    config = load_config()

    experiment_definitions = get_experiment_definitions()

    all_results = execute_experiments(config, experiment_definitions)

    generate_summary(all_results)
    generate_model_comparison(all_results)

    log_program_complete()


if __name__ == '__main__':
    main()