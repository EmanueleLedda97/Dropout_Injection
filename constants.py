import os

# List of dropout rates used for the UCI datasets
# DROPOUT_RATES = [0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.0]
DROPOUT_RATES = [0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# List of z-score percentiles for the normal distribution, from 50 to 95
ZS = [0, 0.126, 0.253, 0.385, 0.524, 0.674, 0.842, 1.036, 1.282, 1.645]
P_SPLITS = 10

PALETTE = {
    'org': '#187498',
    'scl_nll': '#EB5353',
    'scl_cal': '#F7A440',
    'perf': '#36AE7C',

    'emb': '#EB5353',
    'inj': '#187498',

    'emb_opt': '#EB5353',
    'emb_scl': '#F7A440',
    'inj_opt': '#187498',
    'inj_scl': '#36AE7C',

}

# List of the UCI datasets used for the experiment
UCI_DATASETS = ['bostonHousing',
                'concrete',
                'energy',
                'kin8nm',
                'power-plant',
                'protein-tertiary-structure',
                'wine-quality-red',
                'yacht']

# List of UCI datasets divided in two used for a better plot management
UCI_DATASETS_SPLITTED = [['bostonHousing', 'concrete', 'energy', 'kin8nm'],
                         ['power-plant', 'protein-tertiary-structure', 'wine-quality-red', 'yacht']]

# List of mean and variance induced using the training set of each dataset (used for normalize the data in [0,1])
NORMALIZE_DICT = {
    'bostonHousing':                (22.53, 9.19),
    'concrete':                     (35.82, 16.70),
    'energy':                       (22.31, 10.08),
    'kin8nm':                       (0.71, 0.26),
    'naval-propulsion-plant':       (0, 0),
    'power-plant':                  (454.37, 17.07),
    'wine-quality-red':             (5.34, 0.81),
    'protein-tertiary-structure':   (7.74, 6.12),
    'yacht':                        (10.50, 15.14)
}

# Optimal hyperparameters for each experiment
HYPERPARAMETER_CROSS_DICT = {
    'bostonHousing':                (0.001, 128, 300),
    'concrete':                     (0.001, 128, 650),
    'energy':                       (0.001, 64, 1000),
    'kin8nm':                       (0.00003, 64, 300),
    'naval-propulsion-plant':       (0.00004, 64, 140),
    'power-plant':                  (0.00004, 64, 140),
    'wine-quality-red':             (0.00003, 64, 750),
    'protein-tertiary-structure':   (0.00005, 64, 300),
    'yacht':                        (0.0005, 32, 1250)
}

# List of the UCI datasets categorized as "large datasets"
LARGE_DATASETS = ['protein-tertiary-structure']

# List of roots
UCI_ROOT = os.path.join('UCI_Datasets')

# Loading modalities for the dataset
AVAILABLE_DATASET_MODES = ['training', 'test', 'validation']

# Root for saving (eventually) temporary the experiments
TEMP_EXP_FOLDER = os.path.join('experiments', 'temporary')