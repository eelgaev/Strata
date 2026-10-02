// include/strata/platform/thread_affinity.hpp - helper threads leave the host thread's core.
//
// **A LINUX THREAD STARTS WITH ITS CREATOR'S AFFINITY** (Windows gives it the process's).  The session pins the host
// thread to one core (pin_current_thread, pool.hpp), and every thread the host creates after that - the prompt
// path's stager and issuer, the router lookahead, the PLE and direct-file readers - would share that core with it
// and with each other.  On an IBM AC922 (160 logical processors) the prompt path's issuer and the host then handed
// every expert to each other through `yield` on CPU 0, and a 10K prompt left the GPU idle 91% of the time.
//
// pin_current_thread records the affinity from before the first pin and the host's core here; a helper thread calls
// release_inherited_pin() first, and when it carries exactly the host's pin it gets that earlier affinity back.
// Anywhere else it is a no-op.  Header-only, so the platform and reader libraries need not link the CPU pool.
#pragma once

#if defined(_WIN32)
namespace strata::platform {
inline void release_inherited_pin() {}
}  // namespace strata::platform
#else
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <mutex>

namespace strata::platform {

struct HostPin {
    std::mutex mu;
    bool valid = false;
    cpu_set_t before;              // the whole set: a 64-bit mask loses CPUs 64 and up (a POWER9, a big EPYC)
    std::atomic<int> core{-1};
};
inline HostPin& host_pin() {
    static HostPin p;
    return p;
}

/// The host thread was pinned to `core`; `before` is its affinity from before (the first pin's is kept).
inline void note_host_pin(const cpu_set_t& before, int core) {
    HostPin& p = host_pin();
    {
        std::lock_guard<std::mutex> lk(p.mu);
        if (!p.valid) {
            p.before = before;
            p.valid = true;
        }
    }
    p.core.store(core, std::memory_order_release);
}

inline void release_inherited_pin() {
    HostPin& p = host_pin();
    const int core = p.core.load(std::memory_order_acquire);
    if (core < 0) return;
    cpu_set_t cur;
    CPU_ZERO(&cur);
    if (pthread_getaffinity_np(pthread_self(), sizeof cur, &cur) != 0) return;
    if (CPU_COUNT(&cur) != 1 || !CPU_ISSET(core, &cur)) return;   // not the host's pin: leave it
    cpu_set_t to;
    {
        std::lock_guard<std::mutex> lk(p.mu);
        if (!p.valid) return;
        to = p.before;
    }
    pthread_setaffinity_np(pthread_self(), sizeof to, &to);
}

}  // namespace strata::platform
#endif
