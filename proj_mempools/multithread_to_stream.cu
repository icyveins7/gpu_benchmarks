#include "containers/streams.cuh"
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include "pinnedalloc.cuh"
#include <thrust/device_vector.h>

struct TicketLock {
  std::mutex mtx;
  std::condition_variable cv;
  size_t next_ticket = 0;
  size_t serving = 0;

  size_t lock() {
    std::unique_lock<std::mutex> lk(mtx);
    const size_t ticket = next_ticket++;
    printf("Starting to wait for my ticket %zu\n", ticket);
    cv.wait(lk, [&] { return serving == ticket; });
    printf("My ticket %zu has been served\n", ticket);
    lk.unlock();

    return ticket;
  }

  void unlock() {
    std::lock_guard<std::mutex> lk(mtx);
    ++serving;
    cv.notify_all();
  }
};

class ThreadWorker {
public:
  // Delete everything copy/move
  ThreadWorker(const ThreadWorker &) = delete;
  ThreadWorker(ThreadWorker &&) = delete;
  ThreadWorker &operator=(const ThreadWorker &) = delete;
  ThreadWorker &operator=(ThreadWorker &&) = delete;

  static ThreadWorker &instance() {
    static ThreadWorker instance;
    return instance;
  }

  TicketLock &ticketer() { return m_ticketlock; }

  thrust::device_vector<int> &d_input() { return m_d_input; }
  thrust::device_vector<int> &d_output() { return m_d_output; }

  containers::CudaStream &stream() { return m_stream; }

private:
  thrust::device_vector<int> m_d_input;
  thrust::device_vector<int> m_d_output;
  TicketLock m_ticketlock;
  containers::CudaStream m_stream;

  ThreadWorker() {}
  ~ThreadWorker() {}
};

template <typename T> __global__ void dummyKernel(T *buf, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += blockDim.x * gridDim.x) {
    buf[i] += 1;
  }
}

void thread_work(const int *h_input, const size_t len, int *h_output) {
  auto &inst = ThreadWorker::instance();
  auto &stream = inst.stream();
  cudaError_t err;

  // Get a ticket
  size_t ticket = inst.ticketer().lock();
  // --------- CRITICAL SECTION ------------
  // Begin work once your queue is ready
  // Resize required device vectors
  if (inst.d_input().size() < len) {
    printf("Ticket %zu, resizing input to %zu\n", ticket, len);
    inst.d_input().resize(len);
  }
  if (inst.d_output().size() < len) {
    printf("Ticket %zu, resizing output to %zu\n", ticket, len);
    inst.d_output().resize(len);
  }

  // Copy the data in with a stream
  printf("Ticket %zu launch, copying input to device\n", ticket);
  err = cudaMemcpyAsync(inst.d_input().data().get(), h_input, len * sizeof(int),
                        cudaMemcpyHostToDevice, stream());
  if (err != cudaSuccess) {
    throw std::runtime_error("cudaMemcpyAsync failed: " +
                             std::string(cudaGetErrorString(err)));
  }
  printf("Ticket %zu launch, copying input to device done\n", ticket);

  // Run a dummy kernel
  printf("Ticket %zu, launching kernel\n", ticket);
  dummyKernel<<<len / 1024, 1024, 0, stream()>>>(inst.d_input().data().get(),
                                                 len);
  printf("Ticket %zu, launching kernel done\n", ticket);

  // Copy the data out with a stream
  printf("Ticket %zu launch, copying output from device\n", ticket);
  cudaMemcpyAsync(h_output, inst.d_output().data().get(), len * sizeof(int),
                  cudaMemcpyDeviceToHost, stream());
  printf("Ticket %zu launch, copying output from device done\n", ticket);

  // --------- END CRITICAL SECTION ------------

  // Release the ticket
  inst.ticketer().unlock();
  printf("Ticket %zu launches complete\n", ticket);
}

int main(int argc, char **argv) {
  printf("Multithread to stream experiment\n");

  printf("Allocating pinned host vectors..");
  size_t length = 20000000;
  thrust::pinned_host_vector<int> h_input(length);
  thrust::pinned_host_vector<int> h_output(length);
  printf("Done.\n");

  std::thread threads[16];
  for (int i = 0; i < 16; ++i) {
    threads[i] = std::thread(thread_work, h_input.data().get(), length,
                             h_output.data().get());
  }

  printf("Waiting for threads to join...");
  for (int i = 0; i < 16; ++i) {
    threads[i].join();
  }
  printf("Done\n");

  printf("Waiting for all work on stream to complete...");
  ThreadWorker::instance().stream().sync();
  printf("Done.\n");

  return 0;
}
