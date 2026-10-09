// Read feedback only. No motor, target, mode, tool or error-clear API is called.
#include "MarvinSDK.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <iomanip>
#include <thread>
#include <vector>
#include <cstdio>
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t stopping = 0;
void stop(int) { stopping = 1; }
long long mono_ns() {return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();}
long long unix_ns() {return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
template<class T> void array_json(const T* data, int n, double scale=1.0) {
  std::cout << '['; for(int i=0;i<n;i++){if(i)std::cout<<','; double v=data[i]*scale; if(std::isfinite(v))std::cout<<v;else std::cout<<"null";}std::cout<<']';
}
double percentile(std::vector<double> values, double p) {
  if(values.empty()) return 0;
  std::sort(values.begin(),values.end());
  return values[static_cast<size_t>((values.size()-1)*p)];
}
int main(int argc,char**argv) {
  if(argc!=3 && argc!=4) {std::cerr<<"usage: robot_feedback_readonly IPV4 SECONDS [--stream]\n";return 2;}
  bool stream = argc==4;
  if(stream && std::string(argv[3])!="--stream") return 2;
  unsigned a,b,c,d;char extra; double seconds;
  try {seconds=std::stod(argv[2]);}catch(...){return 2;}
  if(std::sscanf(argv[1],"%u.%u.%u.%u%c",&a,&b,&c,&d,&extra)!=4 || a>255||b>255||c>255||d>255 || !std::isfinite(seconds)||seconds<=0||seconds>3600) return 2;
  int byte_order=-1;int compat=CheckSDKTypeCompat(&byte_order);
  if(compat<0) {std::cerr<<"SDK ABI check failed: "<<compat<<"\n";return 3;}
  std::signal(SIGINT,stop);std::signal(SIGTERM,stop);
  if(!OnLinkTo(a,b,c,d)) {std::cerr<<"OnLinkTo failed; no recovery/control attempted\n";return 4;}
  const double rad=std::acos(-1.0)/180;
  const auto start=Clock::now();auto next=start;auto last_sample=start-std::chrono::seconds(1);
  std::array<int,2> last_frame{{-1,-1}};
  std::array<long long,2> last_update{{0,0}},unique{{0,0}},repeated{{0,0}},invalid{{0,0}},nonfinite{{0,0}},error_polls{{0,0}},serial_jumps{{0,0}};
  std::array<std::vector<double>,2> intervals;
  std::vector<double> read_ms;long long polls=0,failed=0,late=0;DCSS feedback{};
  std::cout<<std::setprecision(10);
  while(!stopping && std::chrono::duration<double>(Clock::now()-start).count()<seconds) {
    next+=std::chrono::nanoseconds(3333333);auto begin=Clock::now();
    const bool ok=OnGetBuf(&feedback);const auto received=mono_ns();read_ms.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());polls++;
    if(!ok)failed++;
    if(ok) for(int arm=0;arm<2;arm++) {
      const auto &out=feedback.m_Out[arm];int frame=out.m_OutFrameSerial;
      if(frame<0 || frame>1000000){invalid[arm]++;continue;}
      if(frame==last_frame[arm])repeated[arm]++;
      else {if(last_update[arm])intervals[arm].push_back((received-last_update[arm])/1e6);
        if(last_frame[arm]>=0 && frame<last_frame[arm] && !(last_frame[arm]>900000 && frame<100000))serial_jumps[arm]++;
        unique[arm]++;last_update[arm]=received;last_frame[arm]=frame;}
      bool finite=true;for(int j=0;j<7;j++)finite=finite&&std::isfinite(out.m_FB_Joint_Pos[j])&&std::isfinite(out.m_FB_Joint_Vel[j])&&std::isfinite(out.m_FB_Joint_SToq[j])&&std::isfinite(out.m_EST_Joint_Force[j]);
      if(!finite)nonfinite[arm]++;
      if(feedback.m_State[arm].m_ERRCode!=0)error_polls[arm]++;
    }
    if(ok && (stream || std::chrono::duration<double>(Clock::now()-last_sample).count()>=1)) {
      last_sample=Clock::now();std::cout<<"{\"kind\":\"feedback\",\"schema_version\":1,\"received_monotonic_ns\":"<<received<<",\"received_unix_ns\":"<<unix_ns()<<",\"bias_valid\":false,\"arms\":[";
      for(int arm=0;arm<2;arm++) {if(arm)std::cout<<',';const auto &out=feedback.m_Out[arm];
        std::cout<<"{\"arm\":"<<arm<<",\"frame_serial\":"<<out.m_OutFrameSerial<<",\"state\":"<<feedback.m_State[arm].m_CurState<<",\"error_code\":"<<feedback.m_State[arm].m_ERRCode<<",\"qpos_rad\":";array_json(out.m_FB_Joint_Pos,7,rad);
        std::cout<<",\"qvel_rad_s\":";array_json(out.m_FB_Joint_Vel,7,rad);
        std::cout<<",\"internal_torque_nm\":";array_json(out.m_FB_Joint_SToq,7);
        std::cout<<",\"external_raw_nm\":";array_json(out.m_EST_Joint_Force,7);
        // Read the same parameter echoes used by the collection bridge; never set limits.
        std::cout<<",\"velocity_ratio_percent\":"<<feedback.m_In[arm].m_Joint_Vel_Ratio;
        std::cout<<",\"acceleration_ratio_percent\":"<<feedback.m_In[arm].m_Joint_Acc_Ratio;
        std::cout<<",\"tool_kine\":";array_json(feedback.m_In[arm].m_ToolKine,6);
        std::cout<<",\"tool_dyn\":";array_json(feedback.m_In[arm].m_ToolDyn,10);std::cout<<'}';}
      std::cout<<"]}\n"<<std::flush;
    }
    if(Clock::now()>next){late++;next=Clock::now();}std::this_thread::sleep_until(next);
  }
  bool released=OnRelease();const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
  std::cout<<"{\"kind\":\"summary\",\"elapsed_s\":"<<elapsed<<",\"polls\":"<<polls<<",\"poll_hz\":"<<polls/elapsed<<",\"failed_reads\":"<<failed<<",\"late_polls\":"<<late<<",\"release_ok\":"<<(released?"true":"false")<<",\"read_ms_p99\":"<<percentile(read_ms,.99)<<",\"arms\":[";
  for(int arm=0;arm<2;arm++){if(arm)std::cout<<',';std::cout<<"{\"arm\":"<<arm<<",\"unique_frames\":"<<unique[arm]<<",\"observed_unique_hz\":"<<unique[arm]/elapsed<<",\"repeated_polls\":"<<repeated[arm]<<",\"invalid_serial_polls\":"<<invalid[arm]<<",\"nonfinite_polls\":"<<nonfinite[arm]<<",\"error_polls\":"<<error_polls[arm]<<",\"serial_backwards\":"<<serial_jumps[arm]<<",\"update_interval_ms_p50\":"<<percentile(intervals[arm],.5)<<",\"update_interval_ms_p95\":"<<percentile(intervals[arm],.95)<<",\"update_interval_ms_p99\":"<<percentile(intervals[arm],.99)<<",\"update_interval_ms_max\":"<<percentile(intervals[arm],1)<<'}';}
  std::cout<<"]}\n";return (released && failed==0 && unique[1]>1 && nonfinite[1]==0)?0:5;
}
