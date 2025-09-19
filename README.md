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

# Tips for quick benchmark comparisons

Nsight Systems `nsys` has great command line functionality. Unfortunately, the defaults are very verbose and you often don't want to do anything other than

1. Make a code change for a particular kernel.
2. Recompile and run it again under `nsys profile`.
3. Check the kernel's runtime again.

A very simple process to do the above would be something like this:

```bash
nsys profile --output=benchmark.nsys-rep --force-overwrite=true ./executable && nsys stats --force-export=true --report=cuda_gpu_trace:base benchmark.nsys-rep | grep name_of_kernel
```

Some notes on my choices here:

- We configure the output filename specifically (unless you want to keep all of them, which during development you probably don't). You also need `--force-overwrite` for this.
- We use `cuda_gpu_trace` as it tends to give nicer information. The second number is the duration in nanoseconds. It also splits the individual calls (so you can see if the same kernel has sudden duration changes, depending on whether you expect it).
- `cuda_gpu_trace:base` is to ignore the templating, which is usually very verbose. Don't have to use it if you are using different templates of the same kernel.
