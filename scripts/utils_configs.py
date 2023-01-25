
loan_configs = {
    "data_config": {
        "data_name": "loan",
        "continous_cols": [
            "NoEmp", "NewExist", "CreateJob", "RetainedJob", "DisbursementGross", "GrAppv", "SBA_Appv"
        ],
        "discret_cols": [
            "State", "Term", "UrbanRural", "LowDoc", "Sector_Points"
        ],
        
    },
    "m_config": {
        # model structure    
        "enc_sizes": [50,10],
        "dec_sizes": [10],
        "exp_sizes": [50, 50],
        "sizes": [50, 10, 50],
        "dropout_rate": 0.3,    
        # training module
        'lr': 0.003,
        "lambda_1": 1.0,
        "lambda_3": 0.1,
        "lambda_2": 0.2,
        # adv training
        "epsilon": 0.1,
        "n_steps": 7,
        "k": 2,
        "adv_lr": 0.03
    },
    "t_configs": {
        'n_epochs': 50,
        "batch_size": 256,
        # 'n_epochs': 10,
        'monitor_metrics': 'val/val_loss'
    }, 
    'data_dir_list': [
        f"assets/data/loan/year={year}.csv" for year in range(1994, 2010)
    ]
}

german_configs = {
    "data_config": {
        "data_name": "german_credit",
            "continous_cols": [
            "Duration", "Credit amount", "Installment rate", "Age", "Existing credits", "Number people"
        ],
        "discret_cols": [
            "Present residence", "Status", "History"
        ],
        
    },
    "m_config": {
        # model structure
        "enc_sizes": [100,50],
        "dec_sizes": [20],
        "exp_sizes": [20],
        "dropout_rate": 0.3,
        "sizes": [50, 10, 50],
        # training module
        'lr': 0.003,
        "lambda_1": 1.0,
        "lambda_3": 0.1,
        "lambda_2": 1.0,
        # adv training
        "epsilon": 0.1,
        "n_steps": 10,
        "k": 2,
        "adv_lr": 0.03
    },
    "t_configs": {
        'n_epochs': 100,
        "batch_size": 256,
        # 'n_epochs': 10,
        'monitor_metrics': 'val/val_loss'
    }, 
    'data_dir_list': {
        "assets/data/german_credit/org.csv", "assets/data/german_credit/upt.csv"
    }
}

student_configs = {
    "data_config": {
        "data_name": "student",
        "continous_cols": [
            "failures", "age"
        ],
        "discret_cols": [
            "G2", "G1", "higher", "goout", "Mjob", "Fjob", "health", 
            "freetime", "absences", "Walc", "famrel", "Medu", "Fedu"
        ],
        
    },
    "m_config": {
        # model structure
        "enc_sizes": [50,10],
        "dec_sizes": [10],
        "exp_sizes": [10],
        "sizes": [50, 10, 50],
        "dropout_rate": 0.3,    
        # training module
        'lr': 0.01,
        "lambda_1": 1.0,
        "lambda_3": 0.1,
        "lambda_2": 0.2,
        # adv training
        "epsilon": 0.1,
        "n_steps": 7,
        "k": 2,
        "adv_lr": 0.03
    },
    "t_configs": {
        # 'n_epochs': 50,
        'n_epochs': 100,
        "batch_size": 128,
        # 'n_epochs': 10,
        'monitor_metrics': 'val/val_loss'
    },
    "data_dir_list": [
        "assets/data/student/gp.csv", "assets/data/student/ms.csv"
    ]
}

cov_configs = {
    "data_config": {
        "data_name": "cov",
        "continous_cols": [
            "x1", "x2"
        ],
        "discret_cols": [],
        
    },
    "m_config": {
        # model structure
        "enc_sizes": [50,10],
        "dec_sizes": [10],
        "exp_sizes": [10],
        "sizes": [50, 10, 50],
        "dropout_rate": 0.3,    
        # training module
        'lr': 0.01,
        "lambda_1": 1.0,
        "lambda_3": 0.1,
        "lambda_2": 0.2,
        # adv training
        "epsilon": 0.1,
        "n_steps": 7,
        "k": 2,
        "adv_lr": 0.03
    },
    "t_configs": {
        'n_epochs': 50,
        "batch_size": 128,
        # 'n_epochs': 10,
        'monitor_metrics': 'val/val_loss'
    },
    "data_dir_list": [
        "assets/data/covarient/org.csv", "assets/data/covarient/upt.csv"
    ]
}

label_configs = {
    "data_config": {
        "data_name": "label",
        "continous_cols": [
            "x1", "x2"
        ],
        "discret_cols": [],
    },
    "m_config": {
        # model structure
        "enc_sizes": [50,10],
        "dec_sizes": [10],
        "exp_sizes": [10],
        "sizes": [50, 10, 50],
        "dropout_rate": 0.3,    
        # training module
        'lr': 0.01,
        "lambda_1": 1.0,
        "lambda_3": 0.1,
        "lambda_2": 0.2,
        # adv training
        "epsilon": 0.1,
        "n_steps": 7,
        "k": 2,
        "adv_lr": 0.03
    },
    "t_configs": {
        'n_epochs': 50,
        "batch_size": 128,
        # 'n_epochs': 10,
        'monitor_metrics': 'val/val_loss'
    },
    "data_dir_list": [
        "assets/data/label/org.csv", "assets/data/label/upt.csv"
    ]
}


config_map = {
    'loan': loan_configs,
    'german_credit': german_configs,
    'student': student_configs,
    'covarient': cov_configs,
    'label': label_configs,
}

DATASET_NAMES = config_map.keys() #['loan', 'german_credit', 'student']


def get_configs(data_name):
    if data_name not in DATASET_NAMES:
        raise ValueError(f"`data_name` should be one of `{DATASET_NAMES}`, but got `{data_name}`")
    configs = config_map[data_name]
    
    return (
        configs['m_config'],
        {
            'default_data_config': configs['data_config'],
            't_config': configs['t_configs'],
            'data_dir_list': configs['data_dir_list']
        }
    ) 
