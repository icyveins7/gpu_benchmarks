from typing import Tuple
import numpy as np
from enum import IntEnum

class LookupType(IntEnum):
    POINT = 0
    STRIP_ROW = 1
    STRIP_COL = 2
    RECT = 3

def drawCircle(radiusPixels: int, dtype: type=np.uint32) -> Tuple[np.ndarray, int]:
    length = radiusPixels * 2 + 1
    mat = np.zeros((length, length), dtype=dtype)
    xcoords, ycoords = np.meshgrid(np.arange(length), np.arange(length)) 
    xcoords -= radiusPixels
    ycoords -= radiusPixels

    mask = xcoords**2 + ycoords**2 <= radiusPixels**2
    mat[mask] = 1

    return mat, length

def circleToString(
    mat: np.ndarray,
    lookupTypes: list[LookupType] | None = None,
    lookupResults: list | None = None
) -> str:
    slist = list()

    # Labelled prints
    labelchars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if lookupTypes is not None and lookupResults is not None:
        offset = mat.shape[0] // 2
        ctr = 1
        for _, (section, lookupType) in enumerate(zip(lookupResults, lookupTypes)):
            if lookupType == LookupType.STRIP_ROW:
                # Retrieve row index
                rowIdx = section[0] + offset

                # Retrieve col indices
                colStart = section[2][0] + offset
                colEnd = section[2][1] + offset + 1 # exclusive

                # Amend values
                mat[rowIdx, colStart:colEnd] = ctr
                print("Row %d: %d:%d to %d" % (rowIdx, colStart, colEnd, ctr))

            elif lookupType == LookupType.RECT:
                # Retrieve rows
                rowStart = section[0] + offset
                rowEnd = section[1] + offset + 1 # exclusive

                # Retrieve col indices
                colStart = section[2][0] + offset
                colEnd = section[2][1] + offset + 1 # exclusive

                # Retrieve rows
                mat[rowStart:rowEnd, colStart:colEnd] = ctr
                print("Rect %d:%d: %d:%d to %d" % (rowStart, rowEnd, colStart, colEnd, ctr))

            ctr += 1

    # Simple print
    for row in mat:
        s = "".join(["%c " % labelchars[i % len(labelchars)] if i > 0 else "X " for i in row])
        s += "\n"
        slist.append(s)

    return "".join(slist)

def countNaiveWindow(length: int) -> int:
    return length * 2

def countLookups(lookupTypes: list[LookupType]) -> int:
    counts = np.zeros(len(lookupTypes), np.int32)
    for i, lookupType in enumerate(lookupTypes):
        if lookupType == LookupType.POINT:
            counts[i] += 1
        elif lookupType == LookupType.STRIP_ROW:
            counts[i] += 2
        elif lookupType == LookupType.STRIP_COL:
            counts[i] += 2
        elif lookupType == LookupType.RECT:
            counts[i] += 4

    return int(counts.sum())

def strategy_SATrows(mat: np.ndarray):
    # Needs SAT and row prefixes only

    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square")
    sideLen = mat.shape[0]
    offset = sideLen // 2

    results = list()

    prevStripLen = None
    prevStripBounds = None
    startRow = 0
    for i in range(sideLen//2 + 1): # includes the middle row
        row = mat[i]
        idx = np.argwhere(row == 1).flatten()
        start = int(idx[0]) - sideLen//2
        end = int(idx[-1]) - sideLen//2

        stripLen = end - start + 1
        stripBounds = (start, end)

        print(f"Row {i}: {start}:{end}, length {stripLen}")

        if prevStripLen is not None:
            if prevStripLen != stripLen:
                # Previous one is complete
                results.append((startRow - offset, i - 1 - offset, prevStripBounds))

                # Overwrite with new one
                prevStripBounds = stripBounds
                prevStripLen = stripLen
                startRow = i
        else:
            prevStripBounds = stripBounds
            prevStripLen = stripLen
            startRow = i

    # Dump the last result
    results.append((startRow - offset, sideLen//2-offset, prevStripBounds))

    # Mirror it
    for i in range(len(results)-2, -1, -1): # excludes middle row
        results.append((-results[i][1], -results[i][0], results[i][2]))


    # Classify into strips or rects
    print(results)

    lookupTypes = list()
    for r0, r1, cols in results:
        if r0 == r1:
            if cols[0] == cols[1]:
                lookupTypes.append(LookupType.POINT)
            else:
                lookupTypes.append(LookupType.STRIP_ROW)
        else:
            lookupTypes.append(LookupType.RECT)

    return results, lookupTypes

# def strategy_SATrowscols(mat: np.ndarray):
#     # Needs SAT, rows and cols prefixes
#
#     if mat.shape[0] != mat.shape[1]:
#         raise ValueError("Matrix must be square")
#     sideLen = mat.shape[0]
#
#     results = list()
#
#     # Use prefix rows until the middle, then prefix cols after
#     radius = sideLen // 2
#     prefixSwapIdx = radius // 2
#     # We still check only one side i.e.
#     # for rows we check from top until prefixSwapIdx
#     # for cols we check from left until prefixSwapIdx
#
#     prevStripLen = None
#     prevStripBounds = None
#     for i in range(prefixSwapIdx):
#         row = mat[i]
#         idx = np.argwhere(row == 1).flatten()
#         start = int(idx[0]) - sideLen//2
#         end = int(idx[-1]) - sideLen//2
#
#         stripLen = end - start + 1
#         stripBounds = (start, end)
#
#         print(f"Row {i}: {start}:{end}, length {stripLen}")
#
#         if prevStripLen is not None:
#             if prevStripLen != stripLen:
#                 # Previous one is complete
#                 results.append((i-1, prevStripBounds))
#
#                 # Overwrite with new one
#                 prevStripBounds = stripBounds
#                 prevStripLen = stripLen
#         else:
#             prevStripBounds = stripBounds
#             prevStripLen = stripLen
#
#
#     # Classify into strips or rects
#
#     lookupTypes = list()
#     for i in range(len(results)):
#         if i > 0:
#             numRows = results[i][0] - results[i-1][0]
#         else:
#             numRows = results[i][0] + 1
#
#         if numRows > 1:
#             lookupTypes.append(LookupType.RECT)
#         else:
#             lookupTypes.append(LookupType.STRIP_ROW)
#
#     return results, lookupTypes


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        radius = int(sys.argv[1])
    else:
        radius = 10

    mat, sideLen = drawCircle(radius)
    if sideLen <= 64:
        s = circleToString(mat)
        print(s)

    print(f"=== Strategy: SAT + row prefixes")
    results, lookupTypes = strategy_SATrows(mat)
    print(results)
    print(lookupTypes)
    print(f"Total lookups: {countLookups(lookupTypes)} vs naive {countNaiveWindow(sideLen)}")
    if sideLen <= 64:
        print(circleToString(mat, lookupTypes=lookupTypes, lookupResults=results))

    # print(f"=== Strategy: SAT + row + col prefixes")

