config = {
    'ds': ["EUROSAT-MS-SAR"],  #datasets
    'data_names': {"EUROSAT-MS-SAR": ["MS", "SAR"],},
    'TRAIN_BATCH_SIZE': 128,
    'LEARNING_RATE': 1e-4,
    'LEARNING_RATE_DC': 1e-3,
    'MOMENTUM_EMA': .95,
    'EPOCHS': 200,
    'WARM_UP_EPOCH_EMA': 50,

    'GP_PARAM': 10,
    'DC_PARAM': 0.1,
    'ITER_DC': 10, # Inner iterations for domain critic module

    'TH_FIXMATCH': .95,

    'ITER_DC': 10, # Inner iterations for domain-critic optimization
    'ITER_CLF': 1,  # Inner iterations for classifier optimization
    'ALPHA': 1.,
    #decouple_ds = True
}