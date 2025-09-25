import nsyspy as nsys
from pprint import pprint
import pickle

# All hyperparameters
targets = [
    './wccl_experiment_cuda',
    './wccl_experiment_cuda_useactivesitesinwindow',
    './wccl_experiment_cuda_neighbourchainlocal'
]
# windows = [1, 3, 8, 16]
windows = [1, 16]
# Fix blockwidth to 32, so not shown here
blockheights = [1, 2, 4, 8, 16, 32]
tilewidths = [32, 64]
tileheights = [2, 4, 8, 16, 32, 64]

configurations = list()
for target in targets:
    for window in windows:
        for blockheight in blockheights:
            for tilewidth in tilewidths:
                for tileheight in tileheights:
                    configurations.append({
                        'target': target,
                        'tilewidth': tilewidth,
                        'tileheight': tileheight,
                        'blockwidth': 32,
                        'blockheight': blockheight,
                        'windowhdist': window,
                        'windowvdist': window,
                        'fraction': 0.01, # manually changed
                    })

def makeTarget(config: dict):
    args = [config['target']]
    for k, v in config.items():
        if k != 'target':
            args.append(f"--{k}={v}")
    return args


runner = nsys.Runner()
for configuration in configurations:
    try:
        db = runner.export(runner.profile(
            makeTarget(configuration),
            verbose=True
        ))
        # kernels = db.getKernels("local_connect")
        configuration['dbpath'] = db.dbpath
        # configuration['duration_us'] = kernels[0].duration() / 1e3
        pprint(configuration)
    except Exception as e:
        print(e)


with open('configurations.pkl', 'wb') as f:
    pickle.dump(configurations, f)
