#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

#include <thrust/device_vector.h>

struct TicketLock {
  std::mutex mtx;
  std::condition_variable cv;
  size_t next_ticket = 0;
  size_t serving = 0;

  void lock() {
    std::unique_lock<std::mutex> lk(mtx);
    const size_t ticket = next_ticket++;
    printf("Starting to wait for my ticket %zu\n", ticket);
    cv.wait(lk, [&] { return serving == ticket; });
    printf("My ticket %zu has been served\n", ticket);
    lk.unlock();
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

private:
  thrust::device_vector<int> m_d_input;
  thrust::device_vector<int> m_d_output;
  TicketLock m_ticketlock;

  ThreadWorker() {}
  ~ThreadWorker() {}
};

void thread_work(const int *h_input, const size_t len, int *h_output) {
  auto &inst = ThreadWorker::instance();

  // Get a ticket
  inst.ticketer().lock();
  // --------- CRITICAL SECTION ------------
  // Begin work once your queue is ready
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // --------- END CRITICAL SECTION ------------

  // Release the ticket
  inst.ticketer().unlock();
}

int main(int argc, char **argv) {
  printf("Multithread to stream experiment\n");

  std::thread threads[16];
  for (int i = 0; i < 16; ++i) {
    threads[i] = std::thread(thread_work, nullptr, 0, nullptr);
  }

  for (int i = 0; i < 16; ++i) {
    threads[i].join();
  }

  return 0;
}
