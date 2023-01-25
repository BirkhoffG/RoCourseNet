from rocourse_net.experiment import adversarial_experiment, ExperimentLoggerWanbConfigs
from rocourse_net.module import RoCourseNet
from rocourse_net.methods.roar import ROAR

from relax.import_essentials import *
from relax.methods import VanillaCF, CounterNet
from relax.module import PredictiveTrainingModule
from relax.methods.base import BaseCFModule, BaseParametricCFModule, BasePredFnCFModule
import argparse
from .utils_configs import get_configs, DATASET_NAMES
from jax.config import config
config.parse_flags_with_absl()


name2method = {
    "vanilla": VanillaCF,
    "roar": ROAR,
    "counternet": CounterNet,
    "rocoursenet": RoCourseNet,
}


def main(args):
    logger = ExperimentLoggerWanbConfigs(
        project_name='kdd-rocoursenet',
        user_name='birkhoffg',
        experiment_name=f'{args.method}-{args.data_name}',
    )
    m_configs, aux_configs = get_configs(args.data_name)
    m_configs.update({"random_perturbation": args.random})
    m_configs.update({"adv_lr": args.adv_lr})
    m_configs.update({"lr": args.lr})
    cf_module = name2method[args.method](m_configs)

    if isinstance(cf_module, BaseParametricCFModule):
        pred_module = None
    else:
        pred_module = PredictiveTrainingModule(m_configs)

    cf_results_list, experiment_results = adversarial_experiment(
        **aux_configs,
        pred_training_module=pred_module,
        cf_module=cf_module,
        experiment_logger_configs=logger,
    )

    # m_configs.update({
    #     'n_steps': 1, 'adv_lr': 0.003
    # })
    # cf_results_list, experiment_results = adversarial_experiment_cfnet(
    #     **aux_configs,
    #     training_module=RoCourseNetTrainingModule(m_configs),
    #     experiment_logger_configs=ExperimentLoggerWanbConfigs(
    #         project_name='iclr-rocoursenet',
    #         user_name='birkhoffg',
    #         experiment_name=f'cfnet-{args.data_name}',
    #     )
    # )
    # cf_results_list, experiment_results = adversarial_experiment(
    #     **aux_configs,
    #     training_module=CounterNetTrainingModule(m_configs),
    #     experiment_logger_configs=ExperimentLoggerWanbConfigs(
    #         project_name="iclr-rocoursenet",
    #         user_name="birkhoffg",
    #         experiment_name=f"cfnet-{args.data_name}",
    #     ),
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--data_name", 
        type=str, 
        default="student", 
        choices=DATASET_NAMES
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="rocoursenet",
        choices=name2method.keys(),
    )
    parser.add_argument("--random", type=bool, default=False)
    parser.add_argument("--adv_lr", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()

    main(args)
