from nsyspy import NsysSqlite
import pickle
import pandas as pd
import numpy as np

# Replace this file with whatever you need
configfile = 'configurations.pkl'
with open(configfile, 'rb') as f:
    cfgs = pickle.load(f)

for cfg in cfgs:
    db = NsysSqlite(cfg['dbpath'])
    kernels = db.getKernels(["local", "global"])
    cfg['duration'] = [k.duration() for k in kernels]
    cfg['duration_total'] = np.sum(cfg['duration'])


df = pd.DataFrame(cfgs)

