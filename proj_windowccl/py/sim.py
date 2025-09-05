import wccl
import os
print(os.getcwd())

# # Test a random map
# binary_map = rand_binary_map(2, 3)
# idx_map = make_indices_map(binary_map)
# print(binary_map)
# print(idx_map)
#
# idx_map, numChanges = make_connections(idx_map, 1, 1)
# print(idx_map)

# Test a specific map
# binary_string = [
#     "010001",
#     "100010",
#     "010100",
#     "001000"
# ]
# binary_string = [
#     "010101",
#     "101010"
# ]
binary_string = [
    "000001",
    "000010",
    "000001",
    "101000",
    "010100"
]

binary_map = wccl.parse_string_to_binary_map(binary_string)
print(binary_map)
idx_map = wccl.make_indices_map(binary_map)
print(idx_map)

idx_map, numChanges = wccl.connect(idx_map, 1, 1)
print(numChanges)
