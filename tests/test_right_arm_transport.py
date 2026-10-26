"""Compile/run transport against a nominal in-memory SDK double; no network APIs."""
from dataclasses import replace
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pytest

from force_aware_act.control.command_session import SessionOptions
from force_aware_act.control.goal_selection import select_goal
from force_aware_act.control.sdk_transport import RightArmTransport
from scripts.deployment.build_right_arm_transport import build


HEADER = r'''
#pragma once
struct Out {int m_OutFrameSerial=0; double m_FB_Joint_Pos[7]{}, m_FB_Joint_Vel[7]{}, m_FB_Joint_SToq[7]{}, m_EST_Joint_Force[7]{},m_FB_Joint_Cmd[7]{}; int m_LowSpdFlag=1;};
struct In {int m_ImpType=1,m_Joint_Vel_Ratio=30,m_Joint_Acc_Ratio=30; double m_Joint_K[7]{},m_Joint_D[7]{},m_ToolKine[6]{},m_ToolDyn[10]{};};
struct State {int m_CurState=0,m_ERRCode=0;};
struct DCSS {Out m_Out[2]; In m_In[2]; State m_State[2];};
enum {ARM_STATE_IDLE=0,ARM_STATE_TORQ=3};
extern "C" {
int CheckSDKTypeCompat(int*);
long OnGetSDKVersion();
bool OnLinkTo(int,int,int,int);
bool OnGetBuf(DCSS*);
bool OnRelease();
bool OnClearSet();
bool OnSetSend();
bool OnSetJointLmt_B(int,int);
bool OnSetJointKD_B(double*,double*);
bool OnSetTargetState_B(int);
bool OnSetImpType_B(int);
bool OnSetJointCmdPos_B(double*);
int fake_calls();
int fake_send_calls();
int fake_target_calls();
void fake_target_mode(int);
void fake_fail_next_send();
}
'''
SDK = r'''
#include "MarvinSDK.h"
#include <algorithm>
#include <chrono>
#include <thread>
static DCSS data;
static int calls=0;
static int send_calls=0,target_calls=0,target_mode=0;
static bool fail_send=false;
extern "C" {
int fake_calls(){return calls;}
int fake_send_calls(){return send_calls;}
int fake_target_calls(){return target_calls;}
void fake_target_mode(int x){target_mode=x;}
void fake_fail_next_send(){fail_send=true;}
int CheckSDKTypeCompat(int* x){*x=0;return 1;}
long OnGetSDKVersion(){return 100343013;}
bool OnLinkTo(int,int,int,int){++calls;return true;}
bool OnGetBuf(DCSS* x){++data.m_Out[1].m_OutFrameSerial;*x=data;return true;}
bool OnRelease(){++calls;return true;}
bool OnClearSet(){++calls;return true;}
bool OnSetSend(){++calls;++send_calls;if(fail_send){fail_send=false;return false;}return true;}
bool OnSetJointLmt_B(int v,int a){++calls;data.m_In[1].m_Joint_Vel_Ratio=v;data.m_In[1].m_Joint_Acc_Ratio=a;return true;}
bool OnSetJointKD_B(double*k,double*d){++calls;std::copy(k,k+7,data.m_In[1].m_Joint_K);std::copy(d,d+7,data.m_In[1].m_Joint_D);return true;}
bool OnSetTargetState_B(int state){++calls;data.m_State[1].m_CurState=state;return true;}
bool OnSetImpType_B(int type){++calls;data.m_In[1].m_ImpType=type;return true;}
bool OnSetJointCmdPos_B(double*q){++calls;++target_calls;if(target_mode==1)std::this_thread::sleep_for(std::chrono::milliseconds(25));if(target_mode==2)return false;std::copy(q,q+7,data.m_Out[1].m_FB_Joint_Cmd);std::copy(q,q+7,data.m_Out[1].m_FB_Joint_Pos);return true;}
}
'''


@pytest.fixture
def native_transport(tmp_path):
    if not shutil.which("g++") or not shutil.which("readelf"):
        pytest.skip("native compiler unavailable")
    (tmp_path/"MarvinSDK.h").write_text(HEADER)
    (tmp_path/"sdk.cpp").write_text(SDK)
    sdk = tmp_path/"libMarvinSDK.so"
    subprocess.run(["g++", "-shared", "-fPIC", str(tmp_path/"sdk.cpp"), "-o", str(sdk)], check=True)
    options = replace(SessionOptions(), allow_hardware=True,
                      expected_sdk_sha256=hashlib.sha256(sdk.read_bytes()).hexdigest())
    shim = tmp_path/"transport.so"
    manifest = build(tmp_path, shim, options)
    assert manifest["hardware_connected"] is False and manifest["abi"] == 2
    library = ctypes.CDLL(str(sdk)); library.fake_calls.restype = ctypes.c_int
    assert library.fake_calls() == 0
    transport = RightArmTransport(shim, sdk, options, enable_control=True)
    assert library.fake_calls() == 0  # Constructor only checks version; no connection.
    transport.open("127.0.0.1")  # The test SDK has no sockets or actual hardware implementation.
    try:
        yield transport, library
    finally:
        if transport.connected:
            transport.close()


def test_build_then_nominal_right_packets_units_readback_and_revocation(native_transport):
    transport, library = native_transport
    baseline = json.loads(select_goal("x13y07").controller_baseline_json)
    assert transport.configure(baseline) == "submitted"
    assert transport.enter_impedance() == "submitted"
    target = np.array([.1,-.2,.3,-.4,.5,-.6,.7])
    assert transport.target(target, deadline_ns=time.monotonic_ns()+20_000_000) == "submitted"
    row = transport.read()
    assert row["state"] == 3 and row["impedance_type"] == 1
    assert row["velocity_ratio_percent"] == row["acceleration_ratio_percent"] == 10
    np.testing.assert_allclose(row["qpos_rad"], target, atol=1e-12)
    np.testing.assert_allclose(row["controller_target_rad"], target, atol=1e-12)
    np.testing.assert_allclose(row["joint_stiffness_nm_per_degree"], [2.5,2.5,2.5,2,1.5,1.2,1])
    before = library.fake_calls()
    transport.revoke_targets()
    assert transport.target(target, deadline_ns=time.monotonic_ns()+20_000_000) == "error"
    assert library.fake_calls() == before
    assert transport.servo_off() == "submitted" and transport.read()["state"] == 0
    assert transport.last_servo_off_timing['finished_ns'] >= transport.last_servo_off_timing['started_ns']
    transport.close()
    assert not transport.connected


@pytest.mark.parametrize("stage", ["before_staging", "before_send"])
def test_expired_target_never_sends_or_latches_motion_failure(native_transport, stage):
    transport, library = native_transport
    baseline = json.loads(select_goal("x13y07").controller_baseline_json)
    assert transport.configure(baseline) == "submitted"
    assert transport.enter_impedance() == "submitted"
    calls, sends, setters = library.fake_calls(), library.fake_send_calls(), library.fake_target_calls()
    if stage == "before_send":
        library.fake_target_mode(1)
        deadline = time.monotonic_ns()+20_000_000  # Setter takes >=25ms in memory only.
    else:
        deadline = time.monotonic_ns()-1
    assert transport.target(np.zeros(7), deadline_ns=deadline) == "expired"
    assert library.fake_send_calls() == sends
    assert transport.last_submission_detail["detail"] == f"target_expired_{stage}"
    if stage == "before_staging":
        assert library.fake_calls() == calls and library.fake_target_calls() == setters
    library.fake_target_mode(0)
    assert transport.target(np.zeros(7), deadline_ns=time.monotonic_ns()+50_000_000) == "submitted"
    transport.revoke_targets()
    assert transport.servo_off() == "submitted" and transport.read()["state"] == 0


@pytest.mark.parametrize("stage", ["setter", "send"])
def test_latched_target_error_allows_only_revoked_servo_off(native_transport, stage):
    """Offline regression for the observed stop-path lockout; no robot SDK."""
    transport, library = native_transport
    baseline = json.loads(select_goal("x13y07").controller_baseline_json)
    assert transport.configure(baseline) == "submitted"
    assert transport.enter_impedance() == "submitted"
    if stage == "setter":
        library.fake_target_mode(2)
    else:
        library.fake_fail_next_send()
    assert transport.target(np.zeros(7), deadline_ns=time.monotonic_ns()+50_000_000) == "error"
    assert transport.last_submission_detail["detail"] == ("setter_failed" if stage == "setter" else "set_send_failed")
    calls = library.fake_calls()
    library.fake_target_mode(0)
    assert transport.target(np.zeros(7), deadline_ns=time.monotonic_ns()+50_000_000) == "error"
    assert transport.enter_impedance() == "error"
    assert transport.servo_off() == "error"  # Revocation is still mandatory.
    assert library.fake_calls() == calls
    transport.revoke_targets()
    assert transport.servo_off() == "submitted"
    assert transport.read()["state"] == 0
