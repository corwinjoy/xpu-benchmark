// Simple function timer

#ifndef PI_GPU_TIMER_H
#define PI_GPU_TIMER_H

#include <time.h>
#include <sys/time.h>
#include <stdint.h>

// Get time in microseconds
uint64_t get_posix_clock_time ()
{
    struct timespec ts;

    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t) (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
    else
        return 0;
}

void start_time(uint64_t &timer) {
    timer = get_posix_clock_time();
}

void stop_time(time_t const &timer) {
    uint64_t msec = get_posix_clock_time() - timer;
    printf("Time taken %f seconds\n", (double)msec/1000000.0);
}

#endif //PI_GPU_TIMER_H
