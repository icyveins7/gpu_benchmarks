from nsyspy import NsysSqlite
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

def readout(configfilepath: str):
    dirpath = os.path.dirname(os.path.abspath(configfilepath))
    with open(configfilepath, 'rb') as f:
        cfgs = pickle.load(f)

    for cfg in tqdm(cfgs):
        fulldbpath = os.path.join(dirpath, cfg['dbpath'])
        db = NsysSqlite(fulldbpath)
        kernels = db.getKernels(["local", "global"])

        # Note that this now includes pathcompression and readout kernels (last 2)
        # We just ignore this for now for compute time totals
        # Current runs are local -> global * N -> path -> pathcompress -> readout
        # Faster to assume this than to manually check database for strings
        cfg['duration_local'] = kernels[0].duration
        cfg['duration_global'] = [kernels[k].duration for k in range(1, len(kernels) - 2)]
        cfg['duration_pathcompress'] = kernels[-2].duration
        cfg['duration_readout'] = kernels[-1].duration
        cfg['duration_total'] = cfg['duration_local'] + np.sum(cfg['duration_global'])

    return pd.DataFrame(cfgs)

def read_df(dffilepath: str):
    with open(dffilepath, 'rb') as f:
        df = pickle.load(f)
    return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.close('all')
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configfile", default="configurations.pkl")
    parser.add_argument("-d", "--dffile")
    parser.add_argument("-f", "--fraction", default=0.5, type=float)
    args = parser.parse_args()

    if args.dffile is not None:
        df = read_df(args.dffile)
    else:
        df = readout(args.configfile)
        # Dump it for future
        dirpath = os.path.dirname(os.path.abspath(args.configfile))
        print(f"Dumped dataframe to {dirpath}/configurations.df")
        with open(os.path.join(dirpath, "configurations.df"), 'wb') as f:
            pickle.dump(df, f)

    # Standardise method colours
    methodColours = [(1,0,0), (0,1,0), (0,0,1)]
    methodColourMap = {m: methodColours[i] for i, m in enumerate(df['target'].unique())}

    # Filter by fraction first
    fraction = args.fraction
    dff = df[df['fraction'] == fraction]

    # Hardcoded method to label map
    methodToLabel = {
        "./wccl_experiment_cuda_useactivesitesinwindow": "local union find (with book-keeping)",
        "./wccl_experiment_cuda": "local union find",
        "./wccl_experiment_cuda_neighbourchainlocal": "local neighbour propagation",
    }

    # Assume window is square for now..
    uniqueWindows = dff['windowhdist'].unique()
    bestCount = 20
    for window in uniqueWindows:
        dfs = dff[dff['windowhdist'] == window].sort_values('duration_total')
        print(dfs[:bestCount])

        # === 1. TOTAL TIMING SPLIT === Make pretty bar chart
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.set_title(f"Window {window}x{window}, Fraction = {fraction}")
        ax.set_ylabel("Duration (ms), lower is better")
        ax.set_xlabel(f"Configurations (best {bestCount} out of {len(dfs)})")
        existingLabels = set()
        xtickLabels = []
        for i in range(bestCount):
            row = dfs.iloc[i]
            method = row['target']
            methodColour = methodColourMap[method]
            xtickLabels.append(f"tile {row['tilewidth']}x{row['tileheight']}, block {row['blockwidth']}x{row['blockheight']}")
            btm = 0
            # Parse local kernel first
            label = methodToLabel[method]
            if label in existingLabels:
                label = None
            else:
                existingLabels.add(label)
            ax.bar(
                i,
                row['duration_local']/1e6,
                bottom=btm,
                facecolor=methodColour,
                edgecolor='k',
                label=label,
            )
            btm += row['duration_local']/1e6

            # Now the global kernels
            for j, duration in enumerate(row['duration_global']):
                label = "global union find"
                if label in existingLabels:
                    label = None
                else:
                    existingLabels.add(label)

                ax.bar(
                    i,
                    duration/1e6,
                    bottom=btm,
                    facecolor='y',
                    edgecolor='k',
                    label=label,
                )
                btm += duration/1e6

        ax.set_xticks(range(bestCount))
        ax.set_xticklabels(xtickLabels, rotation=30, ha='right')

        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.splitext(args.configfile)[0] + f"-{window}-{fraction}.png")
        print(methodColourMap)

        # 2. === GLOBAL TIMINGS COMPARISON ====
        tiledimcombis = dfs[['tilewidth','tileheight']].value_counts().reset_index(name='tiledimcount')
        # Select the ones you want to use here
        # comparedcombis = [(32, 2), (32, 8), (32, 32)] # width, height
        comparedcombis = [(64, 8), (64, 32), (64, 64)] # width, height
        dfst = dfs[
            ((dfs['tilewidth'] == 32) & (dfs['tileheight'] == 2)) |
            ((dfs['tilewidth'] == 32) & (dfs['tileheight'] == 8)) |
            ((dfs['tilewidth'] == 32) & (dfs['tileheight'] == 32))
        ].sort_values(['tilewidth','tileheight']) # pyright:ignore
        gfig, gax = plt.subplots(1,len(comparedcombis),sharey=True, figsize=(19.2, 10.8))
        gfig.suptitle(f"Global inter-tile kernel comparisons, Window {window}x{window}, Fraction = {fraction}")
        for i, comparedcombi in enumerate(comparedcombis):
            gax[i].set_title(f"Tile {comparedcombi[0]}x{comparedcombi[1]}")
            gax[i].set_ylabel("Duration (ms), lower is better")
            gax[i].set_xlabel(f"Configurations")

            xtickLabels = []
            dfst = dfs[(dfs['tilewidth'] == comparedcombi[0]) & (dfs['tileheight'] == comparedcombi[1])].sort_values(['blockwidth','blockheight']) # pyright:ignore

            for j in range(len(dfst)):
                btm = 0
                row = dfst.iloc[j]
                label = f"block {row['blockwidth']}x{row['blockheight']}"
                xtickLabels.append(label)
                for duration in row['duration_global']:
                    gax[i].bar(
                        int(j), # pyright:ignore
                        duration,
                        bottom=btm,
                        facecolor='y',
                        edgecolor='k',
                    )
                    btm += duration


            gax[i].set_xticks(range(len(dfst)))
            gax[i].set_xticklabels(xtickLabels, rotation=30, ha='right')

        gfig.tight_layout()
        gfig.savefig(os.path.splitext(args.configfile)[0] + f"-{window}-{fraction}-globalcomparison.png")









