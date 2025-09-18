import numpy as np

def create_neighbour_matrix(input: np.ndarray, hdist: int, vdist: int):
    if input.dtype != np.uint8:
        raise ValueError("input must be uint8")
    if hdist < 0:
        raise ValueError("hdist must be non-negative")
    if vdist < 0:
        raise ValueError("vdist must be non-negative")

    activeIdx = np.argwhere(input)
    alpha = np.eye(activeIdx.shape[0], dtype=np.uint8)

    for i, (row, col) in enumerate(activeIdx):
        # Only need to iterate against the ones after
        for j in range(i, activeIdx.shape[0]):
            otherRow = activeIdx[j, 0]
            otherCol = activeIdx[j, 1]

            # Check window dist
            withinWindow = abs(row - otherRow) <= vdist and abs(col - otherCol) <= hdist

            if withinWindow:
                alpha[i, j] = 1
                alpha[j, i] = 1 # You cannot skip the writes to the transposed index

    return alpha, activeIdx

def create_neighbour_matrix_with_inactive(input: np.ndarray, hdist: int, vdist: int):
    if input.dtype != np.uint8:
        raise ValueError("input must be uint8")
    if hdist < 0:
        raise ValueError("hdist must be non-negative")
    if vdist < 0:
        raise ValueError("vdist must be non-negative")

    # Use zeros instead because we dont want to mark inactive
    alpha = np.zeros((input.size, input.size), dtype=np.uint8)
    beta = np.zeros(input.size, dtype=np.uint8)

    for i in range(input.size):
        row = i // input.shape[1]
        col = i % input.shape[1]
        if input[row, col] == 1:
            beta[i] = 1
        else:
            continue # Skip the inactive ones

        # Only need to iterate against the ones after
        for j in range(col, input.size):
            otherRow = j // input.shape[1]
            otherCol = j % input.shape[1]
            if input[otherRow, otherCol] == 0:
                continue

            # Check window dist
            withinWindow = abs(row - otherRow) <= vdist and abs(col - otherCol) <= hdist

            if withinWindow:
                alpha[i, j] = 1
                alpha[j, i] = 1 # You cannot skip the writes to the transposed index

    return alpha, beta

class NeighbourChainer:
    def __init__(
        self,
        alpha: np.ndarray,
        activeIdx: np.ndarray | None = None,
        beta: np.ndarray | None = None
    ):
        self._alpha = alpha
        self._activeIdx = activeIdx
        self._beta = np.ones(self._alpha.shape[0], dtype=np.uint8) if beta is None else beta

    @property
    def neighbours(self) -> np.ndarray:
        return self._alpha

    @property
    def availability(self) -> np.ndarray:
        return self._beta

    @property
    def isComplete(self) -> bool:
        return bool(np.all(self._beta == 0))

    def extract(self, idx: int) -> np.ndarray:
        """
        Must not overwrite until you are done with the row, as
        numpy will present you with a view i.e. if you 'delete'
        the row by setting to 0 before using it, you will be using a row of 0s.
        """
        self._beta[idx] = 0
        return self._alpha[idx, :]

    def chain(self):
        if self.isComplete:
            print("Already complete.")
            return

        # Take out the next beta index
        bi = np.argwhere(self._beta != 0).flatten()[0]
        # Extract associated row
        seed = self.extract(bi)
        # Extract unconsumed neighbours (gamma)
        ofInterest = np.logical_and(seed, self._beta)

        # Loop over unconsumed neighbours
        while not np.all(ofInterest == 0):
            # print(f"Gamma[{bi}] = ", ofInterest.astype(np.uint8))
            for i, flag in enumerate(ofInterest):
                if flag == 1 and i != bi:
                    # Extract associated row
                    neighbourRow = self.extract(i)
                    # print(f"Extracting row {i} for seed {bi}", neighbourRow.astype(np.uint8))
                    # Combine with seed
                    seed = np.logical_or(seed, neighbourRow)
                    # print(f"Seed[{bi}] is now ", seed.astype(np.uint8))
                    # 'Delete' consumed row
                    self._alpha[i, :] = 0

            # Recalculate neighbours
            ofInterest = np.logical_and(seed, self._beta)

        # Write seed back
        self._alpha[bi, :] = seed

    def readout(self, shape: tuple):
        if not self.isComplete:
            raise ValueError("Not complete, call chain() until completion")

        labels = np.zeros(shape, dtype=np.int32) - 1
        for i, row in enumerate(self._alpha):
            if np.all(row == 0):
                continue
            cluster = row.reshape(shape)
            labels[cluster == 1] = i

        return labels






