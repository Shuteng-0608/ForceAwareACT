"""Pure right-flange equivalent wrench from measured joint disturbance.

q [7] rad, tau [7] Nm -> geometric J [6,7], wrench [6] N/Nm.
No SDK or device access. Shared with the offline force-curve analyzer.
"""
import numpy as np
from .kinematics import _BASE, _FLANGE


def flange_jacobian(q_rad):
    """Base-expressed geometric Jacobian [6,7], reference point right flange."""
    q = np.asarray(q_rad, dtype=np.float64)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("q_rad must contain seven finite joint positions")
    transform = np.eye(4)
    origins, axes = [], []
    for base, angle in zip(_BASE, q):
        transform = transform @ base
        origins.append(transform[:3, 3].copy()/1000.)
        axes.append(transform[:3, 2].copy())
        c, s = np.cos(angle), np.sin(angle)
        rz = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        transform = transform @ rz
    position = (transform @ _FLANGE)[:3, 3]/1000.
    axes, origins = np.asarray(axes), np.asarray(origins)
    return np.concatenate((np.cross(axes,position-origins).T,axes.T),axis=0)


def equivalent_wrench(jacobian, torque_nm, *, length_scale_m=.1, rcond=1e-10):
    """Scaled SVD least squares, retaining residual/rank instead of hiding it.

    Unknown u=[L*F,M] has uniform N*m units; [Jv/L;Jw]^T u ~= tau.
    For full column rank this is the ordinary unweighted joint-torque least
    squares solution. Scaling also makes the condition diagnostic meaningful.
    """
    j = np.asarray(jacobian,dtype=np.float64); tau=np.asarray(torque_nm,dtype=np.float64)
    if j.shape!=(6,7) or tau.shape!=(7,) or not np.isfinite(j).all() or not np.isfinite(tau).all():
        raise ValueError("expected finite Jacobian [6,7] and torque [7]")
    if not np.isfinite(length_scale_m) or length_scale_m<=0 or not np.isfinite(rcond) or not 0<rcond<1:
        raise ValueError("length_scale_m and rcond must be positive and finite")
    scaled=j.copy(); scaled[:3]/=length_scale_m
    u,_,rank,singular=np.linalg.lstsq(scaled.T,tau,rcond=rcond)
    wrench=u.copy(); wrench[:3]/=length_scale_m
    reconstructed=j.T@wrench; residual=tau-reconstructed
    condition=float(singular[0]/singular[-1]) if rank==6 else float('inf')
    return {'wrench_base':wrench,'reconstructed_torque_nm':reconstructed,
            'unexplained_torque_nm':residual,'relative_residual':float(np.linalg.norm(residual)/max(np.linalg.norm(tau),.1)),
            'scaled_condition':condition,'rank':int(rank)}


def rotate_wrench_to_flange(wrench_base,rotation_base_from_flange):
    """Rotate six components at the SAME flange point; no uncalibrated TCP shift."""
    w=np.asarray(wrench_base,dtype=np.float64);r=np.asarray(rotation_base_from_flange,dtype=np.float64)
    if w.shape!=(6,) or r.shape!=(3,3) or not np.isfinite(w).all() or not np.isfinite(r).all():
        raise ValueError("expected finite wrench [6] and rotation [3,3]")
    if not np.allclose(r.T@r,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(r),1.,atol=1e-8):
        raise ValueError("rotation must belong to SO(3)")
    return np.r_[r.T@w[:3],r.T@w[3:]]
