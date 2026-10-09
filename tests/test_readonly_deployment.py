"""Audit helpers and SDK call boundary, without hardware access."""
import importlib.util
from pathlib import Path
import re
import subprocess
import shutil
import pytest
ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "scripts/deployment"
if not DEPLOY.exists():
    DEPLOY = ROOT / "src"
spec = importlib.util.spec_from_file_location("camera_audit", DEPLOY / "audit_realsense_readonly.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_camera_timing_and_gaps():
    rows = [{"received_monotonic_ns": n, "frame_number": f} for n, f in [(0, 10), (33000000, 11), (99000000, 13)]]
    result = module.summarize_frames(rows, .1)
    assert result["frame_number_gaps"] == 1
    assert result["interval_ms_max"] == 66
    assert result["received_hz"] == 30
    assert module.summarize_frames([], 1)["status"] == "no_frames"
    rows.append({"received_monotonic_ns": 100000000, "frame_number": 13})
    assert module.summarize_frames(rows, .1)["nonincreasing_frame_numbers"] == 1


def test_robot_sdk_boundary_and_fake_feedback(tmp_path):
    source = DEPLOY / "robot_feedback_readonly.cpp"
    calls = set(re.findall(r'\b(On\w+|CheckSDKTypeCompat)\s*\(', source.read_text()))
    assert calls == {"OnLinkTo", "OnGetBuf", "OnRelease", "CheckSDKTypeCompat"}
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    (tmp_path / "MarvinSDK.h").write_text('''
#include <cstdlib>
struct Out {int m_OutFrameSerial=0; float m_FB_Joint_Pos[7]{},m_FB_Joint_Vel[7]{},m_FB_Joint_SToq[7]{},m_EST_Joint_Force[7]{};};
struct State {int m_CurState=0,m_ERRCode=0;};
struct In {float m_ToolKine[6]{},m_ToolDyn[10]{}; int m_Joint_Vel_Ratio=10,m_Joint_Acc_Ratio=30;};
struct DCSS {Out m_Out[2]; State m_State[2]; In m_In[2];};
inline int CheckSDKTypeCompat(int* p){*p=0;return 1;}
inline bool OnLinkTo(unsigned char,unsigned char,unsigned char,unsigned char){return true;}
inline bool OnRelease(){return true;}
inline bool OnGetBuf(DCSS* p){static int f=0;++f;for(int a=0;a<2;a++){p->m_Out[a].m_OutFrameSerial=f;p->m_Out[a].m_FB_Joint_Pos[0]=180;p->m_Out[a].m_EST_Joint_Force[0]=2;}return !std::getenv("FAKE_FAIL");}
''')
    binary = tmp_path / "reader"
    subprocess.run([compiler,"-std=c++17","-I",str(tmp_path),str(source),"-o",str(binary)],check=True)
    result = subprocess.run([str(binary),"192.168.1.190","0.03"],capture_output=True,text=True,check=True)
    import json, math, os
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert records[0]["bias_valid"] is False
    assert records[0]["arms"][1]["qpos_rad"][0] == pytest.approx(math.pi)
    assert records[0]["arms"][1]["external_raw_nm"][0] == 2
    assert records[-1]["arms"][1]["unique_frames"] > 1
    failed = subprocess.run([str(binary),"192.168.1.190","0.02"],env={**os.environ,"FAKE_FAIL":"1"},capture_output=True,text=True)
    assert failed.returncode == 5
    assert json.loads(failed.stdout.splitlines()[-1])["release_ok"] is True
    assert subprocess.run([str(binary),"192.168.1.190","nan"],capture_output=True).returncode == 2


def test_causal_availability_has_no_future_frames():
    spec = importlib.util.spec_from_file_location("audit_summary", DEPLOY / "summarize_readonly_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    ticks = list(range(0, 2_000_000_001, 30_000_000))
    report = audit.causal_age_audit(ticks)
    assert report["valid_fraction"] == 1
    assert report["age_ms_max"] < 30
    gap = ticks[:40] + ticks[50:]
    assert audit.causal_age_audit(gap)["valid_fraction"] < 1
    assert audit.causal_age_audit([1, 1])["status"] == "invalid_timestamp_order"


def test_environment_import_has_no_device_side_effects():
    spec = importlib.util.spec_from_file_location("env_check", DEPLOY / "check_tianji_environment.py")
    env = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(env)
    assert callable(env.collect)


def test_software_calibration_rejects_motion_noise_duplicates():
    spec = importlib.util.spec_from_file_location("shadow_readonly", DEPLOY / "shadow_contact_readonly.py")
    shadow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shadow)
    rows = [{"frame_serial":i+1,"received_monotonic_ns":i*100_000_000,
             "qpos_rad":[0.]*7,"qvel_rad_s":[0.]*7,"external_raw_nm":[.3]*7} for i in range(21)]
    value,metrics=shadow.estimate_bias(rows)
    assert value.tolist() == pytest.approx([.3]*7)
    rows[-1]["qvel_rad_s"][0]=.02
    with pytest.raises(ValueError,match="unstable"):
        shadow.estimate_bias(rows)
    rows[-1]["qvel_rad_s"][0]=0
    rows[-1]["frame_serial"]=rows[0]["frame_serial"]
    with pytest.raises(ValueError,match="distinct"):
        shadow.estimate_bias(rows)


def test_checkpoint_checker_import_has_no_hardware_calls():
    spec = importlib.util.spec_from_file_location("checkpoint_check", DEPLOY / "check_contact_checkpoint.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert callable(checker.sha256)
    import hashlib
    assert checker.sha256(DEPLOY / "check_contact_checkpoint.py") == hashlib.sha256((DEPLOY / "check_contact_checkpoint.py").read_bytes()).hexdigest()


def test_rate_limit_uses_full_loop_and_stale_streams():
    spec = importlib.util.spec_from_file_location("rate_shadow", DEPLOY / "shadow_contact_readonly.py")
    shadow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shadow)
    before={"robot_distinct":0,"robot_repeated":0,"robot_stale_ipc":0,"d455":0,"d405":0}
    after={**before,"robot_distinct":3000,"d455":300,"d405":300}
    good=shadow.summarize_rate_stage(50,10,500,500,0,[12]*500,[8]*500,before,after)
    assert good['stable']
    slow=shadow.summarize_rate_stage(100,10,700,700,0,[14]*700,[8]*700,before,after)
    assert not slow['stable']
    assert not slow['checks']['full_loop_p99_within_period']
    stale={**after,"robot_stale_ipc":100}
    assert not shadow.summarize_rate_stage(50,10,500,500,0,[12]*500,[8]*500,before,stale)['stable']
