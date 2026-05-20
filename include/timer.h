#include <chrono>
#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include <atomic>

template <typename Tdur = std::chrono::milliseconds>
class HighResolutionTimer
{
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Event = std::pair<TimePoint, std::string>;

public:
    /**
     * @brief Clears the events/measurements.
     */
    void clear(){
        m_t.clear();
    }

    /**
     * @brief Records a new event/measurement.
     *
     * @param label Optional label for the event.
     * @param enforceFence Adds an atomic signal fence before and after.
     */
    void event(std::string label = "", bool enforceFence = false){
        if (enforceFence)
            std::atomic_signal_fence(std::memory_order_seq_cst);
        m_t.push_back(std::make_pair(std::chrono::high_resolution_clock::now(), label));
        if (enforceFence)
            std::atomic_signal_fence(std::memory_order_seq_cst);
    }

    /**
     * @brief Convenience method to clear() and then immediately record an event().
     *
     * @param label Optional label for the (first) event.
     * @param enforceFence Adds an atomic signal fence before and after.
     */
    void start(std::string label = "start", bool enforceFence = false){
        clear();
        event(label, enforceFence);
    }

    /**
     * @brief Prints a report of all the durations between events.
     */
    void report(){
        for (int i = 1; i < m_t.size(); ++i){
            printSection(m_t.at(i-1), m_t.at(i));
        }
        printTotal();
    }

    /**
     * @brief Convenience method to add a final event() and then report().
     *
     * @param label Optional label for the (last) event.
     * @param enforceFence Adds an atomic signal fence before and after.
     */
    void stop(std::string label = "end", bool enforceFence = false){
        event(label, enforceFence);
        report();
    }

    const std::vector<Event>& measurements(){
        return m_t;
    }

private:
    std::vector<Event> m_t;

    double duration(const TimePoint& t1, const TimePoint &t2){
        using period_t = typename Tdur::period;
        return std::chrono::duration<double, period_t>(t2 - t1).count();
    }
    std::string duration_string();
    void printSection(const Event& t1, const Event& t2){
        printf("%s -> %s : %f %s\n",
               t1.second.c_str(), t2.second.c_str(),
               duration(t1.first, t2.first),
               duration_string().c_str());
    }

    void printTotal(){
        printf("Total %f %s\n",
               duration(m_t.front().first, m_t.back().first),
               duration_string().c_str());
    }
};

template <>
inline std::string HighResolutionTimer<std::chrono::milliseconds>::duration_string(){
    return "ms";
}
template <>
inline std::string HighResolutionTimer<std::chrono::seconds>::duration_string(){
    return "s";
}
