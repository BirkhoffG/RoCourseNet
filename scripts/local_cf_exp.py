import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from rocourset_net.experiment import adversarial_experiment_local_exp, ExperimentLoggerWanbConfigs
from rocourset_net.methods import ROAR
from cfnet.training_module import PredictiveTrainingModule
from cfnet.methods import VanillaCF
from .utils_configs import get_configs, DATASET_NAMES
import argparse


cf_configs = {
    'lr': 0.1, 
    'n_steps': 50, 
    'lambda_': 0.5
}

local_exp_method_dict = {
    'vanilla': VanillaCF,
    'roar': ROAR
}


def main(args):
    m_configs, aux_configs = get_configs(args.data_name)
    aux_configs['t_config']['n_epochs'] = 10
    local_exp_cls = local_exp_method_dict[args.method]
    cf_results_list, experiment_results = adversarial_experiment_local_exp(
        **aux_configs,
        pred_training_module=PredictiveTrainingModule(m_configs),
        cf_moudle=local_exp_cls(cf_configs),
        # experiment_logger_configs=ExperimentLoggerWanbConfigs(
        #     project_name='debug',
        #     user_name='birkhoffg',
        #     experiment_name=args.method,
        # )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', 
                        type=str, 
                        default='loan', 
                        choices=DATASET_NAMES)
    parser.add_argument('--method', 
                        type=str, 
                        default='roar', 
                        choices=local_exp_method_dict.keys())
    args = parser.parse_args()
    
    main(args)