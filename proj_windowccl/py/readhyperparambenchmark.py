from nsyspy import NsysSqlite
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

def readout(configfile: str):
    with open(configfile, 'rb') as f:
        cfgs = pickle.load(f)

    for cfg in tqdm(cfgs):
        db = NsysSqlite(cfg['dbpath'])
        kernels = db.getKernels(["local", "global"])
        cfg['duration'] = [k.duration() for k in kernels]
        cfg['duration_total'] = np.sum(cfg['duration'])

    return pd.DataFrame(cfgs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("configfile", nargs="?", default="configurations.pkl")
    parser.add_argument("--fraction", default=0.5, type=float)
    args = parser.parse_args()
    df = readout(args.configfile)

    # Standardise method colours
    methodColours = [(1,0,0), (0,1,0), (0,0,1)]
    methodColourMap = {m: methodColours[i] for i, m in enumerate(df['target'].unique())}

    fraction = args.fraction

    # Assume window is square for now..
    uniqueWindows = df['windowhdist'].unique()
    bestCount = 20
    for window in uniqueWindows:
        dfs = df[df['windowhdist'] == window].sort_values('duration_total')
        print(dfs[:bestCount])

        # Make pretty bar chart
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.set_title(f"Window {window}x{window}, Fraction = {fraction}")
        ax.set_ylabel("Duration (ns), lower is better")
        ax.set_xlabel(f"Configurations (best {bestCount})")
        existingLabels = set()
        xtickLabels = []
        for i in range(bestCount):
            row = dfs.iloc[i]
            method = row['target']
            methodColour = methodColourMap[method]
            xtickLabels.append(f"tile {row['tilewidth']}x{row['tileheight']}, block {row['blockwidth']}x{row['blockheight']}")
            btm = 0
            for j, duration in enumerate(row['duration']):
                label = method if j == 0 else 'global_union_find'
                if label in existingLabels:
                    label = None
                else:
                    existingLabels.add(label)

                ax.bar(
                    i,
                    duration,
                    bottom=btm,
                    facecolor=methodColour if j == 0 else 'y',
                    edgecolor='k',
                    label=label,
                )
                btm += duration

        ax.set_xticks(range(bestCount))
        ax.set_xticklabels(xtickLabels, rotation=30, ha='right')

        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.splitext(args.configfile)[0] + f"-{window}.png")
        plt.close('all')
        print(methodColourMap)


