"""Offline sensor thresholds and measured-flange endpoints; never loads robot SDK."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scripts.analyze_real_robot_endpoints import flange_fk, load_model


def spans(t, mask, max_gap_s=.02):
    """Observed contiguous spans; gaps never count as sustained evidence."""
    t=np.asarray(t); mask=np.asarray(mask,dtype=bool)
    result=[];start=None;previous=None
    for i,yes in enumerate(mask):
        if start is not None and (not yes or t[i]-t[previous]>max_gap_s):
            result.append((int(start),int(previous),float(t[previous]-t[start])));start=None
        if yes:
            if start is None:start=i
            previous=i
    if start is not None:result.append((int(start),int(previous),float(t[previous]-t[start])))
    return result


def summarize(values):
    a=np.abs(values)
    return {name:np.percentile(a,p,axis=0).tolist() for name,p in [('p99',99),('p999',99.9),('max',100)]}


def threshold_crossings(t, values, limits):
    hit=np.abs(values)>=limits; indices=np.flatnonzero(hit.any(axis=1))
    return dict(triggered=bool(len(indices)),first_s=float(t[indices[0]]) if len(indices) else None,
                first_axes=np.flatnonzero(hit[indices[0]]).tolist() if len(indices) else [],
                longest_s=[max((r[2] for r in spans(t,hit[:,j])),default=0.) for j in range(6)],
                counts=hit.sum(axis=0).tolist())


def analyze(root, split_path, output):
    split=json.loads(Path(split_path).read_text())
    roles={e:role for role in ('train','validation','test') for e in split[role+'_episode_ids']}
    if len(roles)!=55:raise ValueError('expected the reviewed 55 recording split')
    dh=load_model(Path(__file__).resolve().parents[1]/'third_party/tianji_ft/ccs_m6_40.MvKDCfg')
    data=[]
    for name,role in roles.items():
        with h5py.File(Path(root)/name/'episode.hdf5','r') as f:
            if f.attrs['schema_version']!='real_robot_right_arm_raw_v3':raise ValueError('expected raw_v3')
            w=np.asarray(f['observations/ee_wrench']);q=np.asarray(f['observations/joint_pos'])
            ts=np.asarray(f['timestamps/state/log_ns'],dtype=np.int64)
            ft=np.asarray(f['timestamps/ft_sensor/log_ns'],dtype=np.int64)
            t=(ts-ts[0])/1e9;wt=(ft-ts[0])/1e9
            if not np.isfinite(w).all() or not np.isfinite(q).all():raise ValueError('nonfinite inputs')
            xyz=flange_fk(q,dh)[:,:3,3]
            last=t>=t[-1]-.5
            record=dict(episode=name,role=role,source_sha256=f.attrs['source_sha256'],
                duration_s=float(t[-1]),wrench=summarize(w),
                terminal_median_mm=np.median(xyz[last],axis=0).tolist(),
                terminal_spread_mm=float(np.linalg.norm(xyz[last]-np.median(xyz[last],axis=0),axis=1).max()),
                q_min=q.min(axis=0).tolist(),q_max=q.max(axis=0).tolist())
            data.append(dict(record=record,w=w,wt=wt,xyz=xyz,t=t))
    train=[d for d in data if d['record']['role']=='train']
    summary={role:summarize(np.concatenate([d['w'] for d in data if d['record']['role']==role]))
             for role in ('train','validation','test')}
    step=np.array([1,1,1,.05,.05,.01])
    candidates={name:(np.ceil(np.asarray(summary['train'][key])*scale/step)*step).round(6).tolist()
                for name,key,scale in [('train_p99_rounded','p99',1.),('train_p999_margin20','p999',1.2),('train_peak_margin10','max',1.1)]}
    center=np.median([d['record']['terminal_median_mm'] for d in train],axis=0)
    for d in data:
        r=d['record'];dist=np.linalg.norm(d['xyz']-center,axis=1)
        r['terminal_center_distance_mm']=float(np.linalg.norm(np.array(r['terminal_median_mm'])-center))
        r['last5_min_distance_mm']=float(dist[d['t']>=d['t'][-1]-5].min())
        r['goal_radius_checks']={}
        for radius in (3,5):
            runs=spans(d['t'],dist<=radius)
            eligible=[x for x in runs if x[2]>=.5]
            r['goal_radius_checks'][str(radius)]=dict(reached=bool(eligible),
                longest_inside_s=max((x[2] for x in runs),default=0.),
                first_reached_s=float(d['t'][eligible[0][0]]+.5) if eligible else None)
        r['threshold_checks']={name:threshold_crossings(d['wt'],d['w'],values) for name,values in candidates.items()}
    result=dict(data_root=str(Path(root).resolve()),split=str(split_path),roles={r:sum(x==r for x in roles.values()) for r in set(roles.values())},
        units=['N']*3+['N*m']*3,frame='right_ft_measurement',force_statistics=summary,
        threshold_candidates=candidates,threshold_authority='statistical candidates, not hardware safe-load certification',
        goal_center_mm=center.tolist(),goal_method='40 train episode medians of last 0.5s; equal episode weight; fixed before heldout checks',
        all55_terminal_center_mm=np.median([d['record']['terminal_median_mm'] for d in data],axis=0).tolist(),
        goal_scope='measured flange FK; radius and 500ms contiguous dwell; same semantics as sphere monitor, no speed/orientation gate',
        records=[d['record'] for d in data])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze(a.data_root,a.split,a.output)


if __name__=='__main__':main()
