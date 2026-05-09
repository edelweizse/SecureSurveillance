#pragma once

#include <condition_variable>
#include <deque>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>

#include <pipeline/metrics.hpp>

namespace veilsight {
    template <class T>
    class BoundedQueue {
    public:
        explicit BoundedQueue(size_t capacity) : cap_(capacity) {}

        void push_drop_oldest(T v) {
            {
                std::lock_guard lk(m_);
                if (stopped_ || cap_ == 0) return;
                if (q_.size() >= cap_) {
                    q_.pop_front();
                    ++dropped_count_;
                }
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

        void reset() {
            {
                std::lock_guard lk(m_);
                q_.clear();
                dropped_count_ = 0;
                stopped_ = false;
            }
            cv_.notify_all();
        }

        size_t size() const {
            std::lock_guard lk(m_);
            return q_.size();
        }

        size_t capacity() const { return cap_; }

        uint64_t dropped_count() const {
            std::lock_guard lk(m_);
            return dropped_count_;
        }
    private:
        size_t cap_;
        mutable std::mutex m_;
        std::condition_variable cv_;
        std::deque<T> q_;
        bool stopped_ = false;
        uint64_t dropped_count_ = 0;
    };

    template <class T>
    class NamedQueue {
    public:
        NamedQueue(std::string name,
                   std::string producer,
                   std::string consumer,
                   std::string description,
                   size_t capacity)
            : name_(std::move(name)),
              producer_(std::move(producer)),
              consumer_(std::move(consumer)),
              description_(std::move(description)),
              queue_(capacity) {}

        void push_drop_oldest(T v) {
            queue_.push_drop_oldest(std::move(v));
        }

        bool try_pop(T& out) {
            return queue_.try_pop(out);
        }

        bool pop_for(T& out, std::chrono::milliseconds d) {
            return queue_.pop_for(out, d);
        }

        void stop() {
            queue_.stop();
        }

        void reset() {
            queue_.reset();
        }

        size_t size() const {
            return queue_.size();
        }

        size_t capacity() const {
            return queue_.capacity();
        }

        uint64_t dropped_count() const {
            return queue_.dropped_count();
        }

        const std::string& name() const {
            return name_;
        }

        const std::string& producer() const {
            return producer_;
        }

        const std::string& consumer() const {
            return consumer_;
        }

        const std::string& description() const {
            return description_;
        }

        QueueSnapshot snapshot() const {
            QueueSnapshot out;
            out.size = size();
            out.capacity = capacity();
            out.dropped = dropped_count();
            out.producer = producer_;
            out.consumer = consumer_;
            out.description = description_;
            return out;
        }

    private:
        std::string name_;
        std::string producer_;
        std::string consumer_;
        std::string description_;
        BoundedQueue<T> queue_;
    };
}
