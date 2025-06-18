#include <chrono>
#include <iostream>

class HighResolutionTimer
{
public:
    HighResolutionTimer()
    {
        t1 = std::chrono::high_resolution_clock::now();
    }

    ~HighResolutionTimer()
    {
        t2 = std::chrono::high_resolution_clock::now();
        printf("%fs elapsed.\n", std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count());
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t1;
    std::chrono::time_point<std::chrono::high_resolution_clock> t2;
};
