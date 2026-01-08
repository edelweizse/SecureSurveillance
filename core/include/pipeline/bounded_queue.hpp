#pragma once

#include <condition_variable>
#include <deque>
#include <chrono>
#include <mutex>

namespace ss {
    template <class T>
    class BoundedQueue {
    public:
        explicit BoundedQueue(size_t capacity) : cap_(capacity) {}

        void push_drop_oldest(T v) {
            {
                std::lock_guard lk(m_);
                if (stopped_ || cap_ == 0) return;
                if (q_.size() >= cap_) q_.pop_front();
                q_.push_back(std::move(v));
            }
            cv_.notify_one();
        }

        bool try_pop(T& out) {
            std::lock_guard lk(m_);
            if (q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }

        bool pop_for(T& out, std::chrono::milliseconds d) {
            std::unique_lock lk(m_);
            if (!cv_.wait_for(lk, d, [&]{ return stopped_ || !q_.empty(); })) return false;
            if (stopped_ || q_.empty()) return false;
            out = std::move(q_.front());
            q_.pop_front();
            return true;
        }

        void stop() {
            {
                std::lock_guard lk(m_);
                stopped_ = true;
            }
            cv_.notify_all();
        }
    private:
        size_t cap_;
        std::mutex m_;
        std::condition_variable cv_;
        std::deque<T> q_;
        bool stopped_ = false;
    };
}