"""
This script primarily serves as a way to generate some non-uniformly distributed
active pixels inside a matrix. It is meant to be used alongside the built executables.
"""

import numpy as np


def add_circle(row: int, col: int, radius: int, mat: np.ndarray):
    if 2*radius + 1 > mat.shape[0] or 2*radius + 1 > mat.shape[1]:
        raise ValueError("Only allowing radius up to matrix size")

    circle_arr = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    circle_meshx, circle_meshy = np.meshgrid(
        np.arange(-radius, radius + 1),
        np.arange(-radius, radius + 1)
    )
    circle_arr[circle_meshx**2 + circle_meshy**2 <= radius**2] = 1
    count = np.argwhere(circle_arr == 1).size

    startRow = row - radius
    rowOffset = 0
    rowsUsed = 2 * radius + 1
    if startRow < 0: # cut at 0
        rowOffset = -startRow
        rowsUsed += startRow
        startRow = 0

    if startRow + rowsUsed > mat.shape[0]:
        rowsUsed = mat.shape[0] - startRow

    startCol = col - radius
    colOffset = 0
    colsUsed = 2 * radius + 1
    if startCol < 0: # cut at 0
        colOffset = -startCol
        colsUsed += startCol
        startCol = 0

    if startCol + colsUsed > mat.shape[1]:
        colsUsed = mat.shape[1] - startCol

    mat[startRow:startRow + rowsUsed, startCol:startCol + colsUsed] = np.logical_or(
        circle_arr[rowOffset:rowOffset + rowsUsed, colOffset:colOffset + colsUsed],
        mat[startRow:startRow + rowsUsed, startCol:startCol + colsUsed]
    )

    return count

def generate_lognormal_circles(matrows: int, matcols: int, mu: float, sigma: float, fraction: float = 0.5) -> np.ndarray:
    mat = np.zeros((matrows, matcols), dtype=np.uint8)

    # rough estimate of count for log normal
    expectation_radius = np.exp(mu + 0.5 * sigma**2)
    expectation_count_per_circle = np.pi * expectation_radius**2
    rough_num_circles = int(mat.size * fraction / expectation_count_per_circle)
    print(f"Rough number of circles: {rough_num_circles}")
    rough_num_circles *= 2 # just double it to be safe

    # Generate random radii
    radii = np.random.lognormal(mu, sigma, rough_num_circles)
    rows = np.random.randint(0, matrows, rough_num_circles)
    cols = np.random.randint(0, matcols, rough_num_circles)

    # Add circles randomly
    total_count = 0
    for row, col, radius in zip(rows, cols, radii):
        print(f"Adding circle at ({row}, {col}) with radius {int(radius)}")
        count = add_circle(row, col, int(radius), mat)
        total_count += count

        if total_count > mat.size * fraction:
            break

    return mat, total_count


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--mu", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--fraction", type=float)
    # Optional output
    parser.add_argument("--outputdir", type=str, default=None)
    args = parser.parse_args()
    mat, total_count = generate_lognormal_circles(args.rows, args.cols, args.mu, args.sigma, args.fraction)
    if args.rows < 64 and args.cols < 64:
        for row in mat:
            print(row)
    print(f"Total count: {total_count}, {total_count/mat.size * 100}%")

    if args.outputdir is not None:
        path = os.path.join(
            args.outputdir, f"{args.rows}x{args.cols}_mu{args.mu}_sigma{args.sigma}_f{args.fraction}.bin")
        mat.tofile(path)



