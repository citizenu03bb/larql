"""Model-free P1-4 math, written before access to the peer's factors.

No solver or harness imports, no I/O. Inputs are provided only after both
completion gates. Never canonicalize signs or rescale to force agreement.
"""
import numpy as np


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    if a.shape!=b.shape:raise ValueError('Factor shapes differ')
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator)if denominator else None


def compare_factors(ua,va,ub,vb,scale_a=1.,scale_b=1.):
    cu,cv=cosine(ua,ub),cosine(va,vb)
    na=float(abs(scale_a)*np.linalg.norm(np.asarray(ua,dtype=np.float64))*np.linalg.norm(np.asarray(va,dtype=np.float64)))
    nb=float(abs(scale_b)*np.linalg.norm(np.asarray(ub,dtype=np.float64))*np.linalg.norm(np.asarray(vb,dtype=np.float64)))
    if min(na,nb)==0:return dict(cosine_U=cu,cosine_V=cv,norm_a=na,norm_b=nb,agrees=False,reason='undefined zero-norm cosine')
    diff=abs(na-nb);relative=dict(against_a=diff/na,against_b=diff/nb,against_mean=2*diff/(na+nb))
    decisions=[r<=.10 for r in relative.values()];ambiguous=len(set(decisions))>1
    return dict(cosine_U=cu,cosine_V=cv,norm_a=na,norm_b=nb,norm_ratio_max_min=max(na,nb)/min(na,nb),relative_norm_difference=relative,
                norm_convention_ambiguous=ambiguous,agrees=None if ambiguous else bool(cu is not None and cv is not None and cu>=.95 and cv>=.95 and all(decisions)))


def affine_constraint(u,v,key,target,base,scale=1.):
    u,v,key,target,base=[np.asarray(x,dtype=np.float64)for x in (u,v,key,target,base)]
    if u.ndim!=2 or v.ndim!=2 or u.shape[1]!=v.shape[1]or key.shape!=(v.shape[0],)or base.shape!=target.shape or target.shape!=(u.shape[0],):raise ValueError('Affine witness dimensions disagree')
    actual=base+scale*u@(v.T@key);error=actual-target;denominator=np.linalg.norm(target-base)
    return dict(passed=bool(np.allclose(actual,target,atol=2e-5,rtol=2e-5)),max_abs_error=float(np.max(abs(error))),
                error_l2=float(np.linalg.norm(error)),relative_to_update=float(np.linalg.norm(error)/denominator)if denominator else None)
