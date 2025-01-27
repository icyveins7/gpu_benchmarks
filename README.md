# GPU Benchmarks

This is primarily a testing ground for various GPU (currently just CUDA) benchmarks. Here I will try out specific algorithms I have in mind and usually compare them to some of the following:

- Existing GPU library implementations
- Existing CPU library implementations
- Some simple CPU side code

# Prerequisites and Usage

Obviously, you will need CUDA installed. Tests are automatically enabled if you have googletest installed. On Linux, this is easy to install via

```bash
sudo apt install libgtest-dev
```

You will need CMake. Build this just like any other CMake project:

```bash
cmake -B build && cd build
make
```

You should then be able to run the executables inside each subdirectory.
