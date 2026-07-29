#ifndef TZAAR_MCTS_CONCURRENT_QUEUE_H_
#define TZAAR_MCTS_CONCURRENT_QUEUE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace tzaar {

// ─── 執行緒安全的並行佇列 ──────────────────────────────
//
// 使用 mutex + condition_variable 實作，提供 blocking pop 語意。
// 與直接用 std::queue + mutex + cv 不同之處：
//   1. 封裝在單一類別內，外部不需自行管理鎖
//   2. 提供 wait_and_pop() 阻塞直到有元素
//   3. 支援 shutdown 通知（notify_all / reset_stop）
//
// 這個類別是為了取代 SearchManager 中原先的組合：
//   std::queue<int> + std::mutex + std::condition_variable
// 並移除 per-tree mutex（取出即獨佔）。
// ──────────────────────────────────────────────────────────

template<typename T>
class ConcurrentQueue {
 public:
  ConcurrentQueue() = default;
  ~ConcurrentQueue() { stop_ = true; cv_.notify_all(); }

  ConcurrentQueue(const ConcurrentQueue&) = delete;
  ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

  // ─── 生產者 ───────────────────────────────────────
  // 將元素放入佇列，喚醒一個等待中的消費者。
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      queue_.push(std::move(item));
    }
    cv_.notify_one();
  }

  // ─── 非阻塞取出 ───────────────────────────────────
  // 若佇列為空則回傳 false，不回傳 item。
  bool try_pop(T& item) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (queue_.empty()) return false;
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // ─── 阻塞取出 ─────────────────────────────────────
  // 若佇列為空則等待，直到有元素或收到停止訊號。
  // 回傳 true 表示成功取出元素；
  // 回傳 false 表示收到停止訊號（shutdown）。
  bool wait_and_pop(T& item) {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this]() { return !queue_.empty() || stop_; });
    if (stop_ && queue_.empty()) return false;
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // ─── 清空佇列 ─────────────────────────────────────
  void clear() {
    std::lock_guard<std::mutex> lock(mtx_);
    while (!queue_.empty()) queue_.pop();
  }

  // ─── 查詢 ─────────────────────────────────────────
  bool empty() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.empty();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
  }

  // ─── 生命週期控制 ─────────────────────────────────
  // 喚醒所有等待中的 wait_and_pop（用於 shutdown）。
  // 呼叫後 wait_and_pop 會因 stop_=true 而回傳 false。
  void notify_all() {
    stop_ = true;
    cv_.notify_all();
  }

  // 重置 stop flag（用於 reset，讓 wait_and_pop 再次正常等待）。
  void reset_stop() {
    stop_ = false;
  }

 private:
  mutable std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<T> queue_;
  std::atomic<bool> stop_{false};
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_CONCURRENT_QUEUE_H_
