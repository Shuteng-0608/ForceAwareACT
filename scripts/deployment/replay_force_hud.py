"""Replay force HUD from saved logs; never loads cameras or the robot SDK."""
import argparse
import json
from pathlib import Path
import time

from force_aware_act.control.hud_telemetry import HudTelemetry
from force_aware_act.visualization.deployment_hud import (
    DEFAULT_CONFIG, HudProcess, export_replay, read_config, replay_packets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--browser',action='store_true',help='Real-time log playback in local browser')
    parser.add_argument('--port',type=int,default=8787)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if not args.browser:
        print(json.dumps(export_replay(args.run,args.output,config),indent=2))
        return 0
    goal, packets = replay_packets(args.run)
    channel = HudTelemetry()
    hud = HudProcess(channel,config,goal,args.output,args.port)
    print(f'Saved-log replay only: open {hud.url}',flush=True)
    try:
        # Allow time to open the page. Source age is rebased to replay time.
        time.sleep(3)
        origin = time.monotonic_ns()
        historic_origin = packets[0]['record']['received_monotonic_ns']
        for packet in packets:
            source = packet['record']['received_monotonic_ns']
            target = origin+source-historic_origin
            time.sleep(max(0,(target-time.monotonic_ns())/1e9))
            packet['record']['received_monotonic_ns'] = target
            packet['published_ns'] = time.monotonic_ns()
            channel.publish(packet)
        print('Playback complete; retaining STALE display until Ctrl-C.',flush=True)
        while hud.process.is_alive():
            time.sleep(.1)
    except KeyboardInterrupt:
        pass
    finally:
        print(json.dumps(hud.close(),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
