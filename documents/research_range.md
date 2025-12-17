
search_space = {
    'num_conv_layers': (2, 5),                # discrete
    'filters_per_layer': [16, 32, 64, 128],   # discrete choices
    'kernel_size': [3, 5],                     # discrete choices
    'pool_size': [2, 3],                       # discrete choices
    'dropout_rate': (0.0, 0.5),               # continuous
    'dense_units': (64, 512),                 # discrete
    'dense_dropout': (0.0, 0.5),              # continuous
    'learning_rate': (1e-4, 1e-2),            # continuous (log-scale)
    'batch_size': [16, 32, 64, 128],          # discrete choices
    'optimizer': ['adam', 'rmsprop', 'sgd'],  # categorical
    'segment_duration': (1.0, 5.0),           # continuous
    'n_mels': [40, 64, 80, 128],              # discrete choices
    'n_fft': [512, 1024, 2048],               # discrete choices
    'hop_length': [128, 256, 512],            # discrete choices
}