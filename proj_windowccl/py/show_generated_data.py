"""
Primarily convenience script to plot generated images, like those from generate_cluster_data.py
"""

from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

def show_generated_data(filepath: str):
    data = np.fromfile(filepath, np.uint8)
    filename = os.path.splitext(os.path.basename(filepath))[0]
    rows, cols = [int(i) for i in filename.split("_")[0].split("x")]
    data = data.reshape((rows, cols))

    fig, ax = plt.subplots()
    img = ax.imshow(data)

    return fig, ax, img

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    print(f"Loading from {args.filepath}")
    fig, ax, img = show_generated_data(args.filepath)
    if args.save:
        savepath = os.path.splitext(args.filepath)[0] + ".png"
        print(f"Saving to {savepath}")
        fig.tight_layout()
        fig.savefig(savepath)
    else:
        plt.show()
