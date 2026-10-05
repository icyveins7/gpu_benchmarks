# Profiling Repeated CUDA Sections with Nsight Systems

This document summarizes experiments performed with Nsight Systems 2026.1.3 on an NVIDIA L40S. The goal was to capture CUDA kernels, their launching CUDA API calls, and NVTX ranges while avoiding unnecessary CPU and OS tracing.

The example is implemented in `profile_sections_example.cu`. It runs three CUDA sections, each containing 100 kernel launches, with a configurable CPU-only gap before each section.

## Minimal collection options

All experiments use:

```bash
--trace=cuda,nvtx --sample=none --cpuctxsw=none
```

These options retain:

- CUDA runtime and driver API calls.
- CUDA kernels and their correlation to launching API calls.
- CUDA memory operations and synchronization.
- NVTX ranges and markers.

They disable CPU instruction-pointer sampling and CPU context-switch tracing. Explicitly selecting `cuda,nvtx` also excludes OS runtime and OpenGL tracing.

On the tested Nsight Systems installation, plain `nsys profile ./app` defaults to:

```text
--trace=cuda,nvtx,osrt,opengl
--sample=process-tree
--cpuctxsw=process-tree
```

Those defaults can create substantial data unrelated to a CUDA-only investigation.

## Experiment 1: continuous capture

The continuous executable calls `cudaProfilerStart()` before the loop and `cudaProfilerStop()` after it. All three CUDA sections appear in one report. CPU-only gaps are inside the capture.

Run with two-second gaps:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output='./build/profile_sections_2s_%n' \
  ./build/proj_basicops/profile_sections_example 2
```

Run with twenty-second gaps:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output='./build/profile_sections_20s_%n' \
  ./build/proj_basicops/profile_sections_example 20
```

The optional executable argument is the CPU-only gap in seconds. There are three gaps, so these runs capture approximately 6 and 60 seconds respectively.

## Experiment 2: repeated capture using NVTX

The NVTX executable creates a domain named `profile_sections`, registers the string `CUDA section`, and opens one range around each CUDA section.

```bash
sudo nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=nvtx --nvtx-capture='CUDA section@profile_sections' \
  --capture-range-end=repeat --output='./build/profile_sections_nvtx_%n' \
  ./build/proj_basicops/profile_sections_example_nvtx
```

Each range produces a separate `.nsys-rep` file. Repeated capture does not combine disjoint ranges into one timeline.

NVTX capture triggers match registered strings by default. An initial test using only `nvtxRangePushA("CUDA section")` generated no report because that string was not registered. The alternatives are:

- Use a registered NVTX string, as this example does.
- Set `NSYS_NVTX_PROFILER_REGISTER_ONLY=0`, which enables matching unregistered strings but adds matching overhead.

## Experiment 3: repeated capture using the CUDA Profiler API

The CUDA Profiler API executable calls `cudaProfilerStart()` and `cudaProfilerStop()` around every CUDA section:

```bash
sudo nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output='./build/profile_sections_cudaProfilerApi_%n' \
  ./build/proj_basicops/profile_sections_example_cudaProfilerApi
```

The example synchronizes the device before ending each range. Without synchronization, asynchronously launched kernels could still be executing when capture stops. In a real application, prefer an existing synchronization boundary when possible because adding synchronization can alter overlap and performance.

## Sudo requirement observed in this environment

Repeated capture consistently failed without sudo after the first range:

```text
Connection to Agent lost. Internal reason: 'End of file'.
```

The failure affected both CUDA Profiler API and registered-NVTX capture using `--capture-range-end=repeat`. It left a temporary `.qdstrm` but did not generate the requested `.nsys-rep` files.

Using sudo with the same commands allowed repeated capture to generate all three reports. The first sudo attempt also failed once, while a subsequent identical run worked. This suggests a permission, agent initialization, or reconnection issue specific to this Nsight Systems environment; it should not be assumed to affect every system or version.

Continuous single-range capture worked without sudo.

Running Nsight Systems with sudo gives the profiler elevated privileges and may also run the target with elevated privileges. Use it only for trusted executables. Nsight Systems has a `--run-as` option if the profiler must remain privileged while the target should run as another user, but that arrangement was not tested here.

## Measured report sizes

| Capture | Privilege | Contents | Approximate size |
| --- | --- | --- | ---: |
| Continuous, 2-second gaps | User | 300 kernels in one report | 239-240 KiB |
| Continuous, 20-second gaps | User | 300 kernels in one report | 247 KiB |
| Continuous, 2-second gaps | Sudo | 300 kernels in one report | 314 KiB |
| Repeated CUDA Profiler API | Sudo | 100 kernels per report | 292-301 KiB per report |
| Repeated NVTX | Sudo | 100 kernels per report | 291-301 KiB per report |

The non-sudo continuous reports contained:

- 300 kernel instances.
- 300 `cudaLaunchKernel` calls.
- 3 `cudaDeviceSynchronize` calls.
- 3 NVTX ranges.

Each repeated report contained one section:

- 100 kernel instances.
- 100 `cudaLaunchKernel` calls.
- 1 `cudaDeviceSynchronize` call.
- 1 NVTX range.

The separated reports are larger individually even though they contain less CUDA activity. This does not indicate that more CUDA events were captured.

## Why the sizes differ

### Elapsed CPU-only time contributes little

Increasing the three CPU-only gaps from 2 seconds each to 20 seconds each increased capture duration from approximately 6 to 60 seconds. The report grew from 244,761 bytes to 252,667 bytes: an increase of only 7,906 bytes, or 3.23%.

This shows that report growth is primarily driven by recorded event count, not elapsed time, when CPU sampling, context switches, and OS runtime tracing are disabled. A CPU-only gap is represented mostly by timestamps and empty timeline space.

This conclusion changes if an "irrelevant" section performs CUDA API calls, emits many NVTX events, creates processes or threads, or enables another collector. Those events will still consume trace space.

### Every repeated report is self-contained

Each `.nsys-rep` includes fixed metadata such as:

- Report schema and string tables.
- CPU, GPU, kernel, and OS information.
- CUDA device, context, stream, and module descriptions.
- Process and thread metadata.
- Profiler configuration and diagnostics.
- Capability information available to the collector.

A continuous report stores this information once. Three repeated captures store it three times. In this experiment, the three sudo repeated reports total roughly 875-900 KiB, while one sudo continuous report containing all three sections is approximately 314 KiB.

Minor differences between repeated files, such as 292 versus 301 KiB, are expected from initialization, diagnostics, and compression.

### Sudo adds capability metadata

Comparing otherwise equivalent continuous captures showed that sudo increased the compressed report from approximately 240 to 314 KiB.

The largest differences in the exported metadata were:

| Metadata | User capture | Sudo capture |
| --- | ---: | ---: |
| `SupportedFTraceEvents` | 2 bytes | 103,349 bytes |
| `LinuxPerfInfo` | 109 bytes | 17,091 bytes |
| `IsRootEnabled` | Absent | Present |

`SupportedFTraceEvents` is a JSON catalogue of kernel tracepoints visible through Linux tracefs. On this machine it lists 2,095 events across 129 subsystems. Examples include:

- `sched:sched_switch`
- `sched:sched_wakeup`
- `irq:irq_handler_entry`
- `block:block_rq_issue`
- `syscalls:sys_enter_read`
- `power:cpu_frequency`

`LinuxPerfInfo` describes counters and sampling sources available through `perf_event_open()`. On this machine it lists 29 hardware/core events and 9 OS/software events. Examples include:

- CPU and reference cycles.
- Instructions retired.
- Cache and branch misses.
- TLB misses.
- Page faults.
- Context switches and CPU migrations.

It also records low-level event configuration values, the minimum supported sampling period, and the maximum number of simultaneous hardware events.

These fields describe what the privileged profiler could collect. They are not evidence that those events were collected. The sudo report contained no CPU scheduling, sampling, ftrace-event, or perf-event data tables; `--sample=none` and `--cpuctxsw=none` remained effective. For this CUDA-only use case, most of the additional capability metadata is not analytically useful, but Nsight Systems includes it as fixed report metadata.

## Choosing a capture strategy

Prefer continuous capture when:

- The gaps are CPU-only or otherwise produce few traced events.
- All sections should be viewed in one timeline.
- Minimizing total report size and report-generation overhead matters.

Prefer repeated capture when:

- The gaps contain large amounts of irrelevant CUDA activity.
- A separate report per iteration is useful.
- The fixed metadata cost per report is acceptable.
- Sudo is acceptable and required on this environment.

If only a representative sample is needed, capture a small contiguous set of steady-state iterations instead of an entire multi-minute run. Nsight Systems itself warns that long collections can consume substantial storage, and high-frequency CUDA launches can generate large traces even with all unrelated collectors disabled.

## Additional practical details

- `%n` in output names selects the next unused positive integer and avoids overwriting an existing report.
- Report files are written under `./build/` by these commands.
- `nsys stats` exports an additional SQLite database unless one already exists. The SQLite file can be significantly larger than the compressed `.nsys-rep`.
- Write reports to fast local storage rather than network storage when possible.
- CUDA tracing records API calls, kernels, memory operations, and synchronization. A workload launching millions of tiny kernels can therefore produce a large report even during a short capture.
- Use Nsight Systems to identify timeline-level bottlenecks, then use Nsight Compute on selected kernels for detailed kernel analysis.
