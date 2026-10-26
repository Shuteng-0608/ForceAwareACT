#include "tianji_io_teleop_bridge/ft_compensation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <Eigen/Geometry>

namespace tianji_io_teleop_bridge
{
namespace
{
std::string read_file(const std::string & path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {throw std::runtime_error("Cannot read " + path);}
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

template<std::size_t N>
std::array<double, N> numbers(const YAML::Node & node)
{
  if (!node.IsSequence() || node.size() != N) {
    throw std::invalid_argument("Invalid FT calibration array size");
  }
  std::array<double, N> out{};
  for (std::size_t i = 0; i < N; ++i) {
    out[i] = node[i].as<double>();
    if (!std::isfinite(out[i])) {throw std::invalid_argument("Nonfinite FT calibration");}
  }
  return out;
}

template<std::size_t N>
YAML::Node sequence(const std::array<double, N> & values)
{
  YAML::Node node(YAML::NodeType::Sequence);
  for (double x : values) {node.push_back(x);}
  return node;
}
}  // namespace

void FtCompensation::load(const std::string & path, const std::string & model,
  const std::string & validated_model)
{
  loaded_ = false;
  reset();
  const auto text = read_file(path);
  const auto root = YAML::Load(text);
  // The installed model is SHA-256 checked by CMake. Also reject runtime overrides
  // with different bytes, so FK cannot silently use a model unlike the calibration.
  if (root["schema_version"].as<int>() != 1 || root["kind"].as<std::string>() != "real_telemetry" ||
    root["model_sha256"].as<std::string>() !=
    "ce20b1a80974c3ed58cdac4973152625096c8ad93a779e5fd4108c26ce19bd52" ||
    read_file(model) != read_file(validated_model))
  {
    throw std::invalid_argument("FT calibration schema/model mismatch");
  }
  const auto p = root["parameters"], sensor = root["sensor"];
  if (!sensor["axes_confirmed"].as<bool>()) {
    throw std::invalid_argument("FT measurement axes are unconfirmed");
  }
  mass_ = p["mass_kg"].as<double>();
  if (!std::isfinite(mass_) || mass_ <= 0) {throw std::invalid_argument("Invalid FT tool mass");}
  const auto com = numbers<3>(p["com_sensor_m"]), gravity = numbers<3>(root["gravity_base_m_s2"]);
  const auto f = numbers<3>(p["force_bias_N"]), t = numbers<3>(p["torque_bias_Nm"]);
  com_ = Eigen::Map<const Eigen::Vector3d>(com.data());
  gravity_ = Eigen::Map<const Eigen::Vector3d>(gravity.data());
  if (gravity_.norm() < 1e-6) {throw std::invalid_argument("Invalid gravity vector");}
  const auto rotation = sensor["R_flange_sensor"];
  if (!rotation.IsSequence() || rotation.size() != 3) {
    throw std::invalid_argument("Invalid sensor mounting rotation");
  }
  for (std::size_t i = 0; i < 3; ++i) {
    const auto row = numbers<3>(rotation[i]);
    for (std::size_t j = 0; j < 3; ++j) {flange_sensor_(i, j) = row[j];}
    calibration_bias_[i] = f[i];
    calibration_bias_[i + 3] = t[i];
  }
  if (!(flange_sensor_.transpose() * flange_sensor_).isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
    std::abs(flange_sensor_.determinant() - 1.0) > 1e-6)
  {
    throw std::invalid_argument("Sensor rotation must be a proper orthonormal matrix");
  }
  const double fs = sensor["force_sign"].as<double>(), ts = sensor["torque_sign"].as<double>();
  if (std::abs(fs) != 1 || std::abs(ts) != 1) {throw std::invalid_argument("Invalid FT signs");}
  for (std::size_t i = 0; i < 6; ++i) {signs_[i] = i < 3 ? fs : ts;}
  frame_id_ = sensor["frame_id"].as<std::string>();
  if (frame_id_.empty()) {throw std::invalid_argument("Empty FT measurement frame");}
  calibration_file_ = path;
  calibration_text_ = text;
  loaded_ = true;
}

FtCompensation::Wrench FtCompensation::compensate(const Wrench & measured,
  const Eigen::Matrix3d & base_flange) const
{
  if (!loaded_ || !base_flange.allFinite() ||
    !(base_flange.transpose() * base_flange).isApprox(Eigen::Matrix3d::Identity(), 1e-6) ||
    std::abs(base_flange.determinant() - 1.0) > 1e-6)
  {
    throw std::runtime_error("FT compensation unavailable: invalid calibration/FK");
  }
  const Eigen::Vector3d force = mass_ * flange_sensor_.transpose() * base_flange.transpose() * gravity_;
  const Eigen::Vector3d torque = com_.cross(force);
  Wrench result{};
  for (std::size_t i = 0; i < 6; ++i) {
    result[i] = signs_[i] * measured[i] - calibration_bias_[i] - (i < 3 ? force[i] : torque[i - 3]);
    if (!std::isfinite(result[i])) {throw std::runtime_error("Nonfinite compensated FT data");}
  }
  return result;
}

void FtCompensation::observe(double time, int32_t frame, int64_t stamp, int mode, bool low_speed,
  const Joints & q, const Joints & velocity, const Wrench & compensated)
{
  const bool repeated = last_frame_ && *last_frame_ == frame;
  last_frame_ = frame;
  if (!std::isfinite(time) || !low_speed ||
    !std::all_of(q.begin(), q.end(), [](double x) {return std::isfinite(x);}) ||
    !std::all_of(velocity.begin(), velocity.end(), [](double x) {return std::isfinite(x) && std::abs(x) <= 0.2;}) ||
    !std::all_of(compensated.begin(), compensated.end(), [](double x) {return std::isfinite(x);}))
  {
    clear_window();
    return;
  }
  if (!samples_.empty() && (time <= samples_.back().time ||
    time - samples_.back().time > max_gap_sec || mode != samples_.back().mode))
  {
    clear_window();
  }
  if (repeated) {return;}
  samples_.push_back({time, frame, stamp, mode, q, compensated});
  // Retain one sample just before the window boundary to cover a complete window.
  while (samples_.size() > 1 && time - samples_[1].time >= window_sec) {samples_.pop_front();}
}

bool FtCompensation::candidate(double time, Capture & capture, std::string & reason) const
{
  if (!loaded_ || !std::isfinite(time) || samples_.size() < 200 ||
    time < samples_.back().time || time - samples_.back().time > max_gap_sec ||
    samples_.back().time - samples_.front().time < window_sec)
  {
    reason = "Need 5 s (or configured window) of fresh stationary feedback, at least 200 distinct frames";
    return false;
  }
  Capture c;
  c.samples = samples_.size();
  c.span_sec = samples_.back().time - samples_.front().time;
  c.first_stamp_ns = samples_.front().stamp_ns;
  c.last_stamp_ns = samples_.back().stamp_ns;
  c.mode = samples_.back().mode;
  auto qmin = samples_.front().q, qmax = qmin;
  Wrench m2{};
  double count = 0;
  for (const auto & s : samples_) {
    ++count;
    for (std::size_t j = 0; j < 7; ++j) {
      qmin[j] = std::min(qmin[j], s.q[j]);
      qmax[j] = std::max(qmax[j], s.q[j]);
      c.mean_q[j] += (s.q[j] - c.mean_q[j]) / count;
    }
    for (std::size_t i = 0; i < 6; ++i) {
      const double delta = s.compensated[i] - c.bias[i];
      c.bias[i] += delta / count;
      m2[i] += delta * (s.compensated[i] - c.bias[i]);
    }
  }
  for (std::size_t j = 0; j < 7; ++j) {
    if (qmax[j] - qmin[j] > 0.15) {reason = "Joint position changed during FT tare window"; return false;}
  }
  for (std::size_t i = 0; i < 6; ++i) {
    c.stddev[i] = std::sqrt(std::max(0.0, m2[i] / count));
    if (!std::isfinite(m2[i]) || !std::isfinite(c.bias[i]) || c.stddev[i] > (i < 3 ? 0.1 : 0.005)) {
      reason = "FT signal is not stable enough to capture a fixed tare";
      return false;
    }
  }
  capture = c;
  return true;  // No residual-magnitude gate: this is a session zero, not a refit.
}

std::string FtCompensation::save_and_apply(const Capture & capture, const std::string & directory,
  int64_t stamp_ns, const std::string & topic)
{
  if (!loaded_ || directory.empty() || capture.samples < 200 ||
    !std::all_of(capture.bias.begin(), capture.bias.end(), [](double x) {return std::isfinite(x);}))
  {
    throw std::invalid_argument("Invalid FT tare capture or output directory");
  }
  YAML::Node record;
  record["schema_version"] = 1;
  record["kind"] = "right_ft_software_tare";
  record["calibration_file"] = calibration_file_;
  record["calibration_snapshot"] = YAML::Load(calibration_text_);
  record["ros_stamp_ns"] = stamp_ns;
  record["topic"] = topic;
  record["frame_id"] = frame_id_;
  record["hardware_zero_called"] = false;
  record["software_tare_N_Nm"] = sequence(capture.bias);
  record["compensated_std_N_Nm"] = sequence(capture.stddev);
  record["mean_joint_deg"] = sequence(capture.mean_q);
  record["samples"] = capture.samples;
  record["span_sec"] = capture.span_sec;
  record["first_sample_stamp_ns"] = capture.first_stamp_ns;
  record["last_sample_stamp_ns"] = capture.last_stamp_ns;
  record["controller_mode"] = capture.mode;
  record["equation"] = "output = signs*(raw/10000) - gravity_wrench(q) - calibration_bias - software_tare";
  YAML::Emitter yaml;
  yaml.SetDoublePrecision(17);
  yaml << record;
  if (!yaml.good()) {throw std::runtime_error("Cannot serialize FT tare record");}
  std::filesystem::create_directories(directory);
  const auto tick = std::chrono::system_clock::now().time_since_epoch();
  const auto id = std::chrono::duration_cast<std::chrono::nanoseconds>(tick).count();
  const auto path = std::filesystem::path(directory) / ("right_ft_tare_" + std::to_string(id) + ".yaml");
  const std::string temporary = path.string() + ".tmp";
  try {
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(temporary);
    stream << yaml.c_str() << '\n';
    stream.close();
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
  bias_ = capture.bias;  // Replace the previous residual mean; never accumulate twice.
  active_ = true;
  record_file_ = path.string();
  return record_file_;
}

FtCompensation::Wrench FtCompensation::corrected(const Wrench & compensated) const
{
  if (!active_) {throw std::runtime_error("FT software tare has not been captured");}
  auto out = compensated;
  for (std::size_t i = 0; i < 6; ++i) {out[i] -= bias_[i];}
  return out;
}

void FtCompensation::reset()
{
  clear_window();
  last_frame_.reset();
  bias_.fill(0);
  active_ = false;
  record_file_.clear();
}
}  // namespace tianji_io_teleop_bridge
