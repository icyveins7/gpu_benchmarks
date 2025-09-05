"""
Window Connected Component Labeling (WCCL)

The python code here is not meant to be efficient within Python, but rather to mirror the logic
that should be implemented inside the eventual CUDA code. Specific points to note:

- The connections for the index map are made on a copy of the input. This is because in any CUDA implementation,
it will be impossible to ensure the order of the the parent traversal for active sites. As such, there is no way
to ensure that the connections from another element can be seen by the current element. Hence, here in the python
implementation we choose to *ignore* all prior elements' writes by reading from the input but writing to a copy.
"""

import numpy as np

def parse_string_to_binary_map(s: list[str]):
    """
    Generates a binary map from a list of equal length strings

    Parameters
    ----------
    s : list[str]
        List of equal length strings.
        E.g. [
            "010001",
            "100010",
        ]

    Returns
    -------
    binary_map : np.ndarray
        np.uint8 binary map of equal shape
    """
    # Check that each string is equal length
    if not all(len(i) == len(s[0]) for i in s):
        raise ValueError("All strings must be equal length")

    binary_map = np.array([
        [int(i) for i in row]
        for row in s
    ], dtype=np.uint8)
    return binary_map

def rand_binary_map(rows: int, cols: int) -> np.ndarray:
    """
    Generates a random binary map, usually for testing.

    Parameters
    ----------
    rows : int
        Number of rows

    cols : int
        Number of columns

    Returns
    -------
    binary_map : np.ndarray
        np.uint8 binary map of shape (rows, cols)
    """
    return np.random.randint(low=0, high=2, size=(rows, cols)).astype(np.uint8)

def make_indices_map(binary_map: np.ndarray, dtype: type = np.int32) -> np.ndarray:
    """
    Converts the binary map to an index map, used as a starting point for connections;
    Active sites (1) receive their flattened 1D index, while inactive sites (0) receive -1.

    Parameters
    ----------
    binary_map : np.ndarray
        Input binary map.

    dtype : type
        Data type of the index map. Defaults to np.int32.

    Returns
    -------
    idx : np.ndarray
        Index map.
    """
    if binary_map.dtype != np.uint8:
        raise ValueError("binary_map must be uint8")
    if np.any(binary_map > 1):
        raise ValueError("binary_map must contain only 1s or 0s")
    idx = np.arange(binary_map.size).reshape(binary_map.shape).astype(dtype)
    idx[binary_map == 0] = -1
    return idx


def make_connections(idx_map: np.ndarray, hdist: int, vdist: int):
    numChanges = 0
    workspace = idx_map.copy()
    for i in range(0, idx_map.shape[0]):
        for j in range(0, idx_map.shape[1]):
            if (idx_map[i,j] < 0):
                continue
            # Extract window around current point
            minrow = max(0, i - hdist)
            maxrow = min(idx_map.shape[0], i + hdist + 1) # non-inclusive
            mincol = max(0, j - vdist)
            maxcol = min(idx_map.shape[1], j + vdist + 1) # non-inclusive
            # print(f"Index ({i}, {j}) has window [{minrow}, {maxrow}, {mincol}, {maxcol}]")
            window_idx = idx_map[minrow:maxrow, mincol:maxcol]
            # print(window_idx)
            root = np.min(window_idx[window_idx >= 0])

            if (workspace[i,j] != root):
                print(f"Setting idx[{i}, {j}] = {root}")
                workspace[i,j] = root
                numChanges += 1

    return workspace, numChanges

def connect(idx_map: np.ndarray, hdist: int, vdist: int, maxIter: int = 100):
    numChanges_list = []
    numIter = 0
    while numIter < maxIter:
        idx_map, numChanges = make_connections(idx_map, hdist, vdist)
        numChanges_list.append(numChanges)
        if numChanges == 0:
            break

        print(idx_map)
        numIter += 1

    if numChanges_list[-1] != 0:
        raise ValueError(f"Failed to connect indices within {maxIter} iterations")

    return idx_map, numChanges_list

