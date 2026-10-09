"""Offline joint-disturbance curves and equivalent M6 right-flange wrench.

Only saved controller events are read. No control SDK, ROS node or device is
opened. tau [N,7] N*m = J_base(q)^T @ wrench [N,6], with q in radians,
linear J in meters/radian and angular J in radians/radian. The flange-point
least-squares equivalent is not a calibrated peg-tip contact measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from force_aware_act.control.kinematics import right_flange_transform
from force_aware_act.control.wrench_estimation import (
    flange_jacobian, equivalent_wrench, rotate_wrench_to_flange)


def moving_mean(values,sample_count):
    """Centered display-only mean. Output metrics always use unfiltered samples."""
    n=int(sample_count)
    left=n//2;right=n-1-left
    weights=np.ones(n)/n
    return np.column_stack([np.convolve(np.pad(v,(left,right),mode='edge'),weights,mode='valid') for v in np.asarray(values).T])


def analysis_windows(duration_s, width_s=5.):
    """Nonempty runtime intervals, clipped to the actual stop time."""
    if not np.isfinite(duration_s) or duration_s<=0 or not np.isfinite(width_s) or width_s<=0:
        raise ValueError('duration and width must be positive and finite')
    return [(float(a), min(float(a+width_s),float(duration_s)))
            for a in np.arange(0,duration_s,width_s)]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-label',default='模型',help='Verified model name for plot titles')
    parser.add_argument('--show-all-wrench-samples',action='store_true',
                        help='Display all wrench estimates; retain original quality diagnostics')
    parser.add_argument('--stage-marker-seconds',type=float,default=None,
                        help='Optional visual time marker; does not label task phases')
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((args.run/'report.json').read_text()); c=report['controller']
    fb=[e['value'] for e in (json.loads(line) for line in (args.run/'controller/events.jsonl').read_text().splitlines()) if e['kind']=='feedback']
    start=next(x['at_ns'] for x in c['state_history'] if x['state']=='RUNNING');end=c['stop']['requested_ns']
    fb=[f for f in fb if start<=f['received_monotonic_ns']<=end]
    time_ns=np.array([f['received_monotonic_ns'] for f in fb],dtype=np.int64)
    assert np.all(np.diff(time_ns)>0)
    t=(time_ns-start)/1e9
    duration_s=(end-start)/1e9
    plot_title=f"{args.model_label} {c['goal_selection']['goal_id']} {duration_s:.1f}秒"
    q=np.asarray([f['qpos_rad'] for f in fb]); raw=np.asarray([f['external_raw_nm'] for f in fb])
    bias=np.asarray(c['calibration']['bias_nm']);tau=raw-bias
    results=[equivalent_wrench(flange_jacobian(qi),ti) for qi,ti in zip(q,tau)]
    w=np.array([r['wrench_base'] for r in results]); ranks=np.array([r['rank'] for r in results])
    cond=np.array([r['scaled_condition'] for r in results]);res=np.array([r['relative_residual'] for r in results])
    reconstructed=np.array([r['reconstructed_torque_nm'] for r in results])
    transforms=np.array([right_flange_transform(qi) for qi in q])
    local=np.array([rotate_wrench_to_flange(wi,tr[:3,:3]) for wi,tr in zip(w,transforms)])
    fn=np.linalg.norm(w[:,:3],axis=1);mn=np.linalg.norm(w[:,3:],axis=1)
    # Existing ROS HUD numeric quality gates; not physical safety thresholds.
    valid=(ranks==6)&(cond<=200)&(res<=.5)&(fn<=1000)&(mn<=100)
    goal=np.asarray(c['goal_selection']['target_position_m'])
    d=np.linalg.norm(transforms[:,:3,3]/1000.-goal,axis=1)*1000
    np.testing.assert_allclose(np.linalg.norm(local[:,:3],axis=1),fn,atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(local[:,3:],axis=1),mn,atol=1e-10)
    csv=np.column_stack((t,q,raw,tau,w,local,fn,mn,cond,res,valid.astype(int)))
    names=['time_s']+[f'q{j}_rad' for j in range(1,8)]+[f'raw_tau{j}_nm' for j in range(1,8)]+[f'zeroed_tau{j}_nm' for j in range(1,8)]
    names += [f'{frame}_{axis}' for frame in ('base','flange') for axis in ('Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm')]
    names+=['force_norm_N','moment_norm_Nm','scaled_condition','relative_residual','quality_pass']
    assert csv.shape==(len(t),len(names))
    np.savetxt(args.output/'curves.csv',csv,delimiter=',',header=','.join(names),comments='')
    def stats(mask):
        vv=mask&valid
        return {'samples':int(mask.sum()),'quality_pass_samples':int(vv.sum()),
                'joint_torque_median_nm':np.median(tau[mask],axis=0).tolist(),
                'joint_torque_peak_abs_nm':np.abs(tau[mask]).max(axis=0).tolist(),
                'force_base_median_N':np.median(w[vv,:3],axis=0).tolist() if vv.any() else None,
                'moment_base_median_Nm':np.median(w[vv,3:],axis=0).tolist() if vv.any() else None,
                'force_flange_median_N':np.median(local[vv,:3],axis=0).tolist() if vv.any() else None,
                'moment_flange_median_Nm':np.median(local[vv,3:],axis=0).tolist() if vv.any() else None,
                'force_norm_median_N':float(np.median(fn[vv])) if vv.any() else None,
                'force_norm_peak_N':float(fn[vv].max()) if vv.any() else None,
                'moment_norm_peak_Nm':float(mn[vv].max()) if vv.any() else None,
                'relative_residual_p99':float(np.percentile(res[mask],99)),
                'condition_max':float(cond[mask].max())}
    output={'run_id':c['run_id'],'model_label':args.model_label,'duration_s':duration_s,
        'display_masks_failed_quality_samples':not args.show_all_wrench_samples,
        'window_wrench_statistics_scope':'quality-passing samples only',
        'all_sample_wrench_statistics':{
            'force_norm_median_N':float(np.median(fn)),
            'force_norm_peak_N':float(fn.max()),
            'moment_norm_peak_Nm':float(mn.max())},
        'stage_marker_seconds':args.stage_marker_seconds,
        'source':'SDK m_EST_Joint_Force (external_raw_nm), not internal joint torque',
        'zeroing':'subtract per-joint software bias calibrated before THIS run, before inversion',
        'bias_nm':bias.tolist(),'reference_point':'right flange, no calibrated peg-tip TCP',
        'wrench_component_order':'Fx,Fy,Fz,Mx,My,Mz','equation':'tau_est = J_base(q)^T wrench_base + unexplained_torque',
        'conversion':'unweighted torque least squares; SVD with characteristic length 0.1m',
        'sign':'algebraic sign inherited from m_EST_Joint_Force; physical action/reaction sign unverified',
        'residual_normalization':'norm(tau-J^T*w) / max(norm(tau),0.1Nm)',
        'quality_gates':{'rank':6,'max_scaled_condition':200,'max_relative_residual':.5,'max_force_norm_N':1000,'max_moment_norm_Nm':100},
        'display_smoothing_ms':50,'statistics_use_unsmoothed_values':True,'gravity_compensation_independently_confirmed':False,
        'hardware_connected':False,'commands_sent':False,'samples':len(t),'quality_pass_samples':int(valid.sum()),
        'rank_min':int(ranks.min()),'condition_max':float(cond.max()),'relative_residual_max':float(res.max()),
        'windows':{f'{a:g}-{b:g}s':stats((t>=a)&(t<b)) for a,b in analysis_windows(duration_s)
                   if ((t>=a)&(t<b)).any()},
        'last_2s':stats(t>=max(0,duration_s-2)),
        'full_run':stats(np.ones(len(t),dtype=bool)),
        'source_sha256':{name:hashlib.sha256((args.run/name).read_bytes()).hexdigest() for name in ('report.json','controller/events.jsonl')}}
    (args.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    plot_curves(args.output,t,tau,w,local,valid,cond,res,d,c['goal_selection']['config']['limits'],
                title=plot_title,stage_marker_s=args.stage_marker_seconds,
                show_all_wrench_samples=args.show_all_wrench_samples)
    print(json.dumps(output,indent=2))


def plot_curves(output,t,tau,w,local,valid,condition,residual,distance,limits,*,title='模型运行',stage_marker_s=None,show_all_wrench_samples=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for name in ('Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Micro Hei','Droid Sans Fallback'):
        if name in fonts:
            plt.rcParams['font.sans-serif']=[name,'DejaVu Sans'];break
    plt.rcParams['axes.unicode_minus']=False
    samples=max(1,int(round(.05/np.median(np.diff(t)))))
    left=samples//2;right=samples-1-left
    visible=np.ones_like(valid,dtype=bool) if show_all_wrench_samples else valid
    smooth_valid=np.convolve(np.pad(visible,(left,right),mode='edge').astype(int),np.ones(samples,dtype=int),mode='valid')==samples
    wrench_display_label='全部样本，无质量屏蔽' if show_all_wrench_samples else '通过质量筛选的样本'
    def stage(ax):
        if stage_marker_s is not None and 0<=stage_marker_s<t[-1]:
            ax.axvspan(stage_marker_s,t[-1],color='gray',alpha=.08)
            ax.axvline(stage_marker_s,color='gray',ls=':',lw=.8)
        ax.set_xlim(0,t[-1]);ax.grid(alpha=.25)
    smooth=moving_mean(tau,samples)
    fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
    for j,ax in enumerate(axes.flat[:7]):
        ax.plot(t,tau[:,j],alpha=.25,lw=.5,color='tab:blue')
        ax.plot(t,smooth[:,j],lw=1.2,color='tab:blue',label='50 ms均值（仅显示）')
        for sign in (-1,1):
            ax.axhline(sign*limits['torque_soft_nm'][j],color='orange',ls='--',lw=.8,label='soft' if sign==1 else None)
            ax.axhline(sign*limits['torque_hard_nm'][j],color='red',ls=':',lw=.8,label='hard' if sign==1 else None)
        ax.set_title(f'J{j+1} 外力矩估计');ax.set_ylabel('N·m');stage(ax)
        if j==0:ax.legend(fontsize=8,ncol=3)
    ax=axes.flat[7];ax.plot(t,distance,color='black');ax.set_title('法兰距完成点');ax.set_ylabel('mm');stage(ax)
    for ax in axes[-1]:ax.set_xlabel('运行时间 / s')
    fig.suptitle(f'{title}：七关节外力矩估计（已减本次软件零偏）',fontsize=15)
    fig.tight_layout();fig.savefig(output/'joint_external_torque.png',dpi=155);plt.close(fig)
    for values,frame,filename in ((w,'右臂基坐标系','equivalent_wrench_base.png'),(local,'法兰坐标系','equivalent_wrench_flange.png')):
        smooth=moving_mean(values,samples)
        fig,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
        for j in range(3):
            for col,offset,unit in ((0,0,'N'),(1,3,'N·m')):
                ax=axes[j,col];i=j+offset;raw=values[:,i].copy(); sm=smooth[:,i].copy()
                raw[~visible]=np.nan;sm[~smooth_valid]=np.nan
                ax.plot(t,raw,alpha=.25,lw=.5,color='tab:blue');ax.plot(t,sm,lw=1.2,color='tab:blue')
                ax.set_title(('F' if col==0 else 'M')+'xyz'[j]);ax.set_ylabel(unit);stage(ax)
        norm=np.linalg.norm(values[:,:3],axis=1);norm[~visible]=np.nan
        ax=axes[3,0];ax.plot(t,norm,color='tab:purple',lw=.8);ax.set_title('合力模长（未平滑）');ax.set_ylabel('N');stage(ax)
        ax=axes[3,1];ax.plot(t,condition,label='缩放条件数',color='tab:green');ax.set_ylabel('条件数');stage(ax)
        other=ax.twinx();other.plot(t,residual,color='tab:red',alpha=.8,label='相对回代残差');other.axhline(.5,color='tab:red',ls='--',lw=.8);other.set_ylabel('相对残差');other.set_ylim(0,max(.55,float(residual.max())*1.05))
        ax.set_title('数值质量：曲线不等同于标定接触力')
        ax.legend(loc='upper left',fontsize=8);other.legend(loc='upper right',fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('运行时间 / s')
        fig.suptitle(f'{title}：等效法兰受力（{frame}表达；{wrench_display_label}）',fontsize=14)
        fig.tight_layout();fig.savefig(output/filename,dpi=155);plt.close(fig)

    smooth=moving_mean(local,samples);smooth[~smooth_valid]=np.nan
    raw=local.copy();raw[~visible]=np.nan
    colors=['tab:blue','tab:orange','tab:green']
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for j,axis in enumerate('xyz'):
        for ax,offset,letter in ((axes[0],0,'F'),(axes[1],3,'M')):
            ax.plot(t,raw[:,j+offset],color=colors[j],alpha=.18,lw=.5)
            ax.plot(t,smooth[:,j+offset],color=colors[j],lw=1.2,label=letter+axis)
    axes[0].set_ylabel('等效力 / N');axes[0].set_title('法兰坐标系：Fx、Fy、Fz（细线原值，粗线50 ms均值）')
    axes[1].set_ylabel('等效力矩 / N·m');axes[1].set_title('法兰坐标系：Mx、My、Mz')
    norm=np.linalg.norm(local[:,:3],axis=1);norm[~visible]=np.nan
    axes[2].plot(t,norm,color='tab:purple',lw=.8,label='合力模长（原值）');axes[2].set_ylabel('N')
    axes[3].plot(t,distance,color='black',label='法兰距完成点');axes[3].set_ylabel('mm');axes[3].set_xlabel('运行时间 / s')
    for ax in axes:
        stage(ax);ax.legend(loc='upper right',fontsize=9,ncol=3)
    axes[3].set_title('法兰距完成点：与受力曲线使用相同时间轴')
    fig.suptitle(f'{title}：等效法兰受力概览（{wrench_display_label}）',fontsize=15)
    fig.tight_layout();fig.savefig(output/'equivalent_wrench_overview.png',dpi=155);plt.close(fig)


if __name__=='__main__':
    main()
