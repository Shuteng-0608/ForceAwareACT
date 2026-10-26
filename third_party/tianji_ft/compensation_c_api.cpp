// Thin C ABI around the unchanged acquisition compensation/tare implementation.
// No ROS, robot SDK, network or actuator API.
#include "tianji_io_teleop_bridge/ft_compensation.hpp"
#include <algorithm>
#include <memory>
#include <string>

using FT = tianji_io_teleop_bridge::FtCompensation;
namespace { thread_local std::string error; }
extern "C" {
const char* ft_error() { return error.c_str(); }
void* ft_create(const char* calibration, const char* model, double window, double gap) {
  try {
    auto ft = std::make_unique<FT>();
    ft->load(calibration, model, model);
    ft->window_sec = window; ft->max_gap_sec = gap;
    return ft.release();
  } catch (const std::exception& e) { error=e.what(); return nullptr; }
}
void ft_destroy(void* handle) { delete static_cast<FT*>(handle); }
void ft_reset(void* handle) { static_cast<FT*>(handle)->reset(); }
int ft_process(void* handle, const double* measured, const double* rotation,
               double* compensated, double* corrected) {
  try {
    auto& ft=*static_cast<FT*>(handle); FT::Wrench w{};
    std::copy_n(measured,6,w.begin());
    Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>> r(rotation);
    auto c=ft.compensate(w,r); std::copy(c.begin(),c.end(),compensated);
    if (!ft.active()) return 0;
    auto z=ft.corrected(c); std::copy(z.begin(),z.end(),corrected); return 1;
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
void ft_observe(void* handle, double time, int frame, long long stamp, int mode,
                int low_speed, const double* qdeg, const double* vdeg, const double* compensated) {
  FT::Joints q{},v{}; FT::Wrench w{};
  std::copy_n(qdeg,7,q.begin()); std::copy_n(vdeg,7,v.begin()); std::copy_n(compensated,6,w.begin());
  static_cast<FT*>(handle)->observe(time,frame,stamp,mode,low_speed!=0,q,v,w);
}
int ft_capture(void* handle, double time, long long stamp, const char* directory,
               double* bias, double* stddev, double* count_span) {
  try {
    auto& ft=*static_cast<FT*>(handle); FT::Capture capture;
    if (!ft.candidate(time,capture,error)) return 0;
    ft.save_and_apply(capture,directory,stamp,"/io_teleop/right/ft_sensor");
    std::copy(capture.bias.begin(),capture.bias.end(),bias);
    std::copy(capture.stddev.begin(),capture.stddev.end(),stddev);
    count_span[0]=static_cast<double>(capture.samples); count_span[1]=capture.span_sec;
    return 1;
  } catch (const std::exception& e) { error=e.what(); return -1; }
}
}
