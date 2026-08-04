#ifndef TZAAR_MCTS_DEBUG_LOG_H_
#define TZAAR_MCTS_DEBUG_LOG_H_

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

namespace tzaar {

// 可開關的執行緒安全 debug logger。
//
// 設計重點：
//   - enabled_ 用 relaxed atomic，off 時呼叫端只做一次 load 即跳過
//   - 真正的格式化與檔案寫入只發生在 enabled_ = true 時
//   - 內部持有常駐 ofstream，避免每次開檔（debug 期間的檔寫效能）
//   - 供 SearchManager 的多執行緒（worker / main / result_handler）共用

class DebugLogger {
 public:
  static DebugLogger& instance() {
    static DebugLogger inst;
    return inst;
  }

  void set_enabled(bool on) { enabled_.store(on, std::memory_order_relaxed); }

  void set_path(const std::string& path) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (path == path_) return;
    path_ = path;
    reopen_locked();
  }

  // off 時回傳 true，讓呼叫端在進入前就跳過（零格式化成本）。
  bool likely_disabled() const {
    return !enabled_.load(std::memory_order_relaxed);
  }

  void log(const char* fmt, ...) {
    if (likely_disabled()) return;  // 關閉 → 直接返回

    std::lock_guard<std::mutex> lk(mtx_);
    if (!enabled_.load(std::memory_order_relaxed)) return;
    if (!out_.is_open()) reopen_locked();
    if (!out_.is_open()) return;

    char buf[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    out_ << buf << '\n';
    out_.flush();  // debug log 需要即時可見
  }

 private:
  DebugLogger() = default;
  ~DebugLogger() {
    if (out_.is_open()) out_.flush();
  }

  DebugLogger(const DebugLogger&) = delete;
  DebugLogger& operator=(const DebugLogger&) = delete;

  void reopen_locked() {
    if (out_.is_open()) out_.close();
    out_.open(path_, std::ios::app);
  }

  std::atomic<bool> enabled_{false};
  std::mutex mtx_;
  std::ofstream out_;
  std::string path_{"logs/cpp_debug.log"};
};

}  // namespace tzaar

#endif  // TZAAR_MCTS_DEBUG_LOG_H_
