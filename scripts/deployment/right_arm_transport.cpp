// Narrow C ABI, single connection owner. Loading this library does not connect.
// Only the right arm (SDK B) is addressable. No tool/error-clear/left-arm API.
#include "MarvinSDK.h"
#include <cmath>
#include <cstddef>
#include <chrono>
#include <thread>

namespace {
bool connected = false, authorized = false, revoked = false, failed = false;
const char* packet_detail = "not_attempted";
using Clock = std::chrono::steady_clock;
Clock::time_point next_api{};
long long monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
template<class F> auto api(F fn) -> decltype(fn()) {
  // Validated vendor header requires >=1 ms between API calls.
  std::this_thread::sleep_until(next_api);
  auto result=fn(); next_api=Clock::now()+std::chrono::milliseconds(1);
  return result;
}
// ABI 2: 1=submitted, 0=busy, 2=expired (no send), -1=error/disabled.
int begin_packet(bool stopping = false) {
  if (!connected) { packet_detail="not_connected"; return -1; }
  if (!authorized) { packet_detail="not_authorized"; return -1; }
  // A failure latches motion off, but must not block the separately authorized
  // IDLE packet after targets are revoked and Python observes standstill.
  if (failed && !stopping) { packet_detail="sdk_failure_latched"; return -1; }
  if (revoked && !stopping) { packet_detail="targets_revoked"; return -1; }
  if (!api(OnClearSet)) { packet_detail="clear_set_busy"; return 0; }
  return 1;
}
int finish_packet(bool staged, long long deadline_ns = -1) {
  if (!staged) {
    (void)api(OnClearSet); // Discard local staging; NEVER OnSetSend after a failed setter.
    failed = true;
    packet_detail="setter_failed";
    return -1;
  }
  bool expired=false;
  bool submitted=api([&]{
    if (deadline_ns>=0 && monotonic_ns()>deadline_ns) { expired=true; return false; }
    return OnSetSend();
  });
  if (expired) {
    // Remove the unsent target before any future packet; never retry it later.
    if (!api(OnClearSet)) {
      failed=true; packet_detail="expired_staging_clear_failed"; return -1;
    }
    packet_detail="target_expired_before_send";
    return 2;
  }
  if (!submitted) { failed = true; packet_detail="set_send_failed"; return -1; }
  packet_detail="submitted";
  return 1; // Submission only. Readback/physical acknowledgement is separate.
}
bool finite7(const double* p) {
  if (!p) return false;
  for (int j=0; j<7; ++j) if (!std::isfinite(p[j])) return false;
  return true;
}
}
extern "C" {
int fa_abi_version() { return 2; }
const char* fa_last_packet_detail() { return packet_detail; }
long fa_sdk_version() { return api(OnGetSDKVersion); }
int fa_open(int a, int b, int c, int d, int explicit_authorization) {
  if (explicit_authorization != 1 || connected || a<0 || a>255 || b<0 || b>255 || c<0 || c>255 || d<0 || d>255) return -1;
  int order=-1;
  if (api([&]{ return CheckSDKTypeCompat(&order); })<0) return -1;
  if (!api([&]{ return OnLinkTo(a,b,c,d); })) return -1;
  connected=true; authorized=true; revoked=false; failed=false;
  return 1;
}
int fa_read(double* values, int length) {
  if (!connected || !values || length != 72) return -1;
  DCSS f{};
  if (!api([&]{ return OnGetBuf(&f); })) return -1;
  const auto &o=f.m_Out[1]; const auto &i=f.m_In[1]; const auto &s=f.m_State[1];
  values[0]=o.m_OutFrameSerial; values[1]=s.m_CurState; values[2]=s.m_ERRCode;
  values[3]=i.m_ImpType; values[4]=i.m_Joint_Vel_Ratio; values[5]=i.m_Joint_Acc_Ratio;
  values[6]=o.m_LowSpdFlag;
  const double rad=std::acos(-1.)/180.;
  for (int j=0;j<7;++j) {
    values[7+j]=o.m_FB_Joint_Pos[j]*rad; values[14+j]=o.m_FB_Joint_Vel[j]*rad;
    values[21+j]=o.m_FB_Joint_SToq[j]; values[28+j]=o.m_EST_Joint_Force[j];
    values[35+j]=i.m_Joint_K[j]; values[42+j]=i.m_Joint_D[j];
    values[65+j]=o.m_FB_Joint_Cmd[j]*rad;
  }
  for (int j=0;j<6;++j) values[49+j]=i.m_ToolKine[j];
  for (int j=0;j<10;++j) values[55+j]=i.m_ToolDyn[j];
  for (int j=0;j<72;++j) if (!std::isfinite(values[j])) return -1;
  return 1;
}
int fa_configure(int velocity, int acceleration, double* k, double* d) {
  if (!finite7(k) || !finite7(d) || velocity<1 || velocity>100 || acceleration<1 || acceleration>100) return -1;
  for (int j=0;j<7;++j) if (k[j]<=0 || d[j]<=0) return -1;
  int result=begin_packet(); if (result!=1) return result;
  return finish_packet(api([&]{ return OnSetJointLmt_B(velocity,acceleration); }) && api([&]{ return OnSetJointKD_B(k,d); }));
}
int fa_enter_impedance() {
  int result=begin_packet(); if (result!=1) return result;
  // Vendor bridge uses a pure mode packet, separate from targets/parameters.
  return finish_packet(api([]{ return OnSetTargetState_B(ARM_STATE_TORQ); }) && api([]{ return OnSetImpType_B(1); }));
}
int fa_target(const double* radians, long long deadline_ns) {
  if (!finite7(radians) || deadline_ns<0) { packet_detail="invalid_target"; return -1; }
  if (!connected || !authorized || failed || revoked) {
    packet_detail="target_permission_denied"; return -1;
  }
  if (monotonic_ns()>deadline_ns) { packet_detail="target_expired_before_staging"; return 2; }
  double degrees[7];
  for (int j=0;j<7;++j) degrees[j]=radians[j]*180./std::acos(-1.);
  int result=begin_packet(); if (result!=1) return result;
  return finish_packet(api([&]{ return OnSetJointCmdPos_B(degrees); }), deadline_ns);
}
void fa_revoke_targets() { revoked=true; }
int fa_servo_off() {
  if (!revoked) { packet_detail="servo_off_requires_revocation"; return -1; }
  int result=begin_packet(true); if (result!=1) return result;
  return finish_packet(api([]{ return OnSetTargetState_B(ARM_STATE_IDLE); }));
}
int fa_close() {
  revoked=true; authorized=false;
  if (!connected) return 1;
  connected=false;
  return api(OnRelease) ? 1 : -1; // Disconnect is NOT a physical stop operation.
}
}
