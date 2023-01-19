import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from rocourset_net.experiment import adversarial_experiment_cfnet, ExperimentLoggerWanbConfigs
from rocourset_net.training_module import RoCourseNetTrainingModule
from cfnet.training_module import CounterNetTrainingModule
from cfnet.import_essentials import *
import argparse
from .utils_configs import get_configs, DATASET_NAMES


# data_config = {
#     "data_name": "loan",
#     "continous_cols": [
#         "NoEmp", "NewExist", "CreateJob", "RetainedJob", "DisbursementGross", "GrAppv", "SBA_Appv"
#     ],
#     "discret_cols": [
#         "State", "Term", "UrbanRural", "LowDoc", "Sector_Points"
#     ],
#     "batch_size": 128,
# }

# data_config = {
#     "data_name": "german_credit",
#     "continous_cols": [
#         "Duration", "Credit amount", "Installment rate", "Age", "Existing credits", "Number people"
#     ],
#     "discret_cols": [
#         "Present residence", "Status", "History"
#     ],
#     "batch_size": 256,
# }
# m_config = {
#     # model structure
#     "enc_sizes": [200,50],
#     "dec_sizes": [50],
#     "exp_sizes": [50],
#     "dropout_rate": 0.3,    
#     # training module
#     'lr': 0.003,
#     "lambda_1": 1.0,
#     "lambda_3": 0.1,
#     "lambda_2": 0.2,
#     # adv training
#     "epsilon": 0.1,
#     "n_steps": 7,
#     "k": 2,
#     "adv_lr": 0.03
# }
# t_configs = {
#     'n_epochs': 50,
#     # 'n_epochs': 10,
#     'monitor_metrics': 'val/val_loss'
# }


# if __name__ == "__main__":
    # with jax.disable_jit():
    #     adversarial_experiment(
    #         training_module=RoCourseNetTrainingModule(m_config),
    #         default_data_config=data_config,
    #         data_dir_list=[ 
    #             f"assets/data/loan/year={year}.csv" for year in range(1994, 2010) 
    #         ],
    #         t_config=t_configs,
    #         experiment_logger_configs=ExperimentLoggerWanbConfigs(
    #             project_name='debug',
    #             user_name='birkhoffg',
    #             experiment_name='counternet_adv',
    #         )
    #     )

    # cf_results_list, experiment_results = adversarial_experiment_cfnet(
    #     training_module=RoCourseNetTrainingModule(m_config),
    #     default_data_config=data_config,
    #     data_dir_list=[ 
    #         f"assets/data/loan/year={year}.csv" for year in range(1994, 2010) 
    #     ],
    #     t_config=t_configs,
    #     experiment_logger_configs=ExperimentLoggerWanbConfigs(
    #         project_name='debug',
    #         user_name='birkhoffg',
    #         experiment_name='counternet_adv',
    #     )
    # )

    # cf_results_list, experiment_results = adversarial_experiment(
    #     training_module=CounterNetTrainingModule(m_config),
    #     default_data_config=data_config,
    #     data_dir_list=[ 
    #         f"assets/data/loan/year={year}.csv" for year in range(1994, 2010) 
    #     ],
    #     t_config=t_configs,
    #     experiment_logger_configs=ExperimentLoggerWanbConfigs(
    #         project_name='debug',
    #         user_name='birkhoffg',
    #         experiment_name='counternet',
    #     )
    # )

def search_params(args):
    m_configs, aux_configs = get_configs(args.data_name)
    # lambda_3_list = np.arange(0.3, 1.1, 0.2)
    lambda_3_list = np.arange(1.1, 2.2, 0.2)
    # lambda_3_list = [0.5]
    # lambda_3_list = [0.1]
    # epsilons = [0.1, 0.01, 0.5]
    # epsilons = [0.1]
    epsilons = [0.05, 0.1]

    for lambda_3 in lambda_3_list:
        for eps in epsilons:
            m_configs.update({
                'epsilon': eps, 'lambda_3': lambda_3
            })
            cf_results_list, experiment_results = adversarial_experiment_cfnet(
                **aux_configs,
                training_module=RoCourseNetTrainingModule(m_configs),
                experiment_logger_configs=ExperimentLoggerWanbConfigs(
                    project_name='tradeoff',
                    user_name='birkhoffg',
                    experiment_name=f'cfnet-{args.data_name}',
                )
            )


def main(args):
    m_configs, aux_configs = get_configs(args.data_name)
    m_configs.update({
        'random_perturbation': args.random
    })
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
    cf_results_list, experiment_results = adversarial_experiment_cfnet(
        **aux_configs,
        training_module=CounterNetTrainingModule(m_configs),
        experiment_logger_configs=ExperimentLoggerWanbConfigs(
            project_name='iclr-rocoursenet',
            user_name='birkhoffg',
            experiment_name=f'cfnet-{args.data_name}',
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', 
                        type=str, 
                        default='loan', 
                        choices=DATASET_NAMES)
    parser.add_argument('--random', 
                        type=bool, 
                        default=False)
    args = parser.parse_args()
    
    main(args)
