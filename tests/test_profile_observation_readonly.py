"""Timing attribution stays independent of camera, CUDA and robot SDK imports."""
from scripts.deployment.profile_observation_readonly import intervals_ms, overlap_ns, summarize


def test_overlap_distinguishes_gc_inside_gap_from_after_gap():
    assert overlap_ns(10, 110, 20, 100) == 80
    assert overlap_ns(10, 110, 110, 150) == 0
    assert overlap_ns(10, 110, 0, 120) == 100


def test_summary_attributes_receive_gap_without_claiming_hardware_drop():
    def camera(at, number):
        return {"name": "d455", "received_ns": at, "frame_number": number,
                "device_ms": number*33.333,
                **{k: 100 for k in ("poll_ns", "copy_ns", "buffer_wait_ns", "buffer_append_ns", "publish_ns")}}
    collection = {"generation": 2, "start_ns": 20_000_000, "end_ns": 90_000_000,
                  "duration_ns": 70_000_000, "collected": 0}
    trace = {"camera": [camera(0, 1), camera(100_000_000, 4)], "feedback": [],
             "inference": [], "gc": [collection], "errors": []}
    report = summarize(trace, [])
    assert report["robot_control_packets"] == report["robot_connections"] == 0
    gap = report["d455"]["gaps_above_50ms"][0]
    assert gap["interval_ms"] == 100
    assert gap["frame_number_delta"] == 3
    assert gap["gc_overlaps"] == [collection]
    assert report["gc_summary"]["2"]["max_ms"] == 70
    assert intervals_ms([0, 20_000_000, 40_000_000])["max"] == 20
