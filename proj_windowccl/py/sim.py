import wccl

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
# binary_string = [
#     "000001",
#     "000010",
#     "000001",
#     "101000",
#     "010100"
# ]
# binary_string = [
#     "01010101010101010101",
#     "10101010101010101010",
#     "01010101010101010101",
#     "10101010101010101010",
#     "01010101010101010101",
#     "10101010101010101010",
#     "01010101010101010101",
# ]
# binary_string = [
#     "00000000010000000100",
#     "00000000101000001000",
#     "00000001000100010000",
#     "00000010000010100000",
#     "00000000000001000000",
#     "00001000000000100000",
#     "00010000000000010000",
# ]
binary_string = [
    "1000001000",
    "0100010100",
    "0010100010",
    "0001000001",
]
# binary_string = [
#     "0001000001",
#     "0010100010",
#     "0100010100",
#     "1000001000",
# ]
binary_string = [
    "1000001",
    "0100010",
    "0010100",
    "0001000",
    "0010100",
    "0100010",
    "1000001",
]
# binary_string = [
#     "1101",
#     "1101",
#     "1101",
# ]
# binary_string = [
#     "0001",
#     "0010",
#     "0100",
#     "1000",
# ]

for reach in range(-1, 0):
    print(f"=============== Reach: {reach}")
    binary_map = wccl.parse_string_to_binary_map(binary_string)
    # print(binary_map)
    idx_map = wccl.make_indices_map(binary_map)
    # print(idx_map)

    idx_map, numChanges = wccl.connect(idx_map, 1, 1,
                                       # reach=-1,
                                       verbose=True)
    print(numChanges)

    wccl.pprint(idx_map)
