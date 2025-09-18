from binaryneighbours import *
import numpy as np
import wccl

binary_string = [
    "10001",
    "01010",
    "00100",
]
binary_string = [
    "1000001",
    "0100010",
    "0010100",
    "0001000",
    "0010100",
    "0100010",
    "1000001",
]
binary_string = [
    "0001000001",
    "0010100010",
    "0100010100",
    "1000001000",
]
binary_string = [
    "0001000001",
    "0010000010",
    "0100010100",
    "1000001000",
]

input = wccl.parse_string_to_binary_map(binary_string)
print(input)

# neighbours, activeIdx = create_neighbour_matrix(input, 1, 1)
# print(activeIdx)
# print(neighbours)
# chainer = NeighbourChainer(neighbours, activeIdx)

neighbours, beta = create_neighbour_matrix_with_inactive(input, 1, 1)
print(neighbours)
print(beta)
chainer = NeighbourChainer(neighbours, beta=beta)

while not chainer.isComplete:
    print("---------------------------")
    chainer.chain()
    print(chainer.neighbours)
    print(chainer.availability)

labels = chainer.readout(input.shape)
wccl.pprint(labels)
