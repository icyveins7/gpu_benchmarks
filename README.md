# GPU Benchmarks

This is primarily a testing ground for various GPU (currently just CUDA) benchmarks. Here I will try out specific algorithms I have in mind and usually compare them to some of the following:

- Existing GPU library implementations
- Existing CPU library implementations
- Some simple CPU side code
- Some existing Python (usually NumPy) implementations

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

# Documentation for subprojects

Sometimes, I may use small python scripts to try to plot illustrations which I find to be helpful in understanding a particular algorithm, or the way a unit test is set up.
These generally shouldn't require any fancy packages other than `matplotlib` and `numpy`, and I will try to keep these in the `/doc/` subdirectory in each of the project directories.

Also, project-specific documentation will be found in the subdirectory's own `README.md`, so as to not make this file too crowded.
