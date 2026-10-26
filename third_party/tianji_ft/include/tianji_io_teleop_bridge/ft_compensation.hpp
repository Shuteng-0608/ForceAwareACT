#ifndef TIANJI_IO_TELEOP_BRIDGE__FT_COMPENSATION_HPP_
#define TIANJI_IO_TELEOP_BRIDGE__FT_COMPENSATION_HPP_

#include <array>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

namespace tianji_io_teleop_bridge
{

// Sensor-frame processing only. This class has no robot-control APIs.
class FtCompensation
{
public:
  using Wrench = std::array<double, 6>;
  using Joints = std::array<double, 7>;
  struct Sample
  {
    double time;
    int32_t frame;
    int64_t stamp_ns;
    int mode;
    Joints q;
    Wrench compensated;
  };
  struct Capture
  {
    Wrench bias{}, stddev{};
    Joints mean_q{};
    std::size_t samples{};
    double span_sec{};
    int64_t first_stamp_ns{}, last_stamp_ns{};
    int mode{};
  };

  void load(const std::string & calibration_file, const std::string & model_file,
    const std::string & validated_model_file);
  Wrench compensate(const Wrench & measured_si, const Eigen::Matrix3d & base_flange) const;
  void observe(double time, int32_t frame, int64_t stamp_ns, int mode, bool low_speed,
    const Joints & q_deg, const Joints & velocity_deg_s, const Wrench & compensated);
  bool candidate(double time, Capture & capture, std::string & reason) const;
  // Save the complete calibration snapshot and capture before activating a new tare.
  std::string save_and_apply(const Capture & capture, const std::string & directory,
    int64_t ros_stamp_ns, const std::string & topic);
  Wrench corrected(const Wrench & compensated) const;
  void clear_window() {samples_.clear();}
  void reset();
  bool active() const {return active_;}
  const Wrench & bias() const {return bias_;}
  const std::string & frame_id() const {return frame_id_;}
  const std::string & record_file() const {return record_file_;}
  const std::string & calibration_file() const {return calibration_file_;}
  double window_sec{5.0};
  double max_gap_sec{0.2};

private:
  bool loaded_{false}, active_{false};
  double mass_{0.0};
  Eigen::Vector3d com_, gravity_;
  Eigen::Matrix3d flange_sensor_;
  Wrench calibration_bias_{}, signs_{}, bias_{};
  std::string calibration_file_, calibration_text_, frame_id_, record_file_;
  std::deque<Sample> samples_;
  std::optional<int32_t> last_frame_;
};

}  // namespace tianji_io_teleop_bridge
#endif
