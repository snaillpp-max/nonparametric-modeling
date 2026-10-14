"""Connected Q1 cavity, reflection reduction, and original coupled RK4 kernels.

Adapted from the verified expanded-range experiment; no private runtime paths.
"""
import numpy as np
from scipy.sparse import coo_matrix,diags
from numba import njit,prange

def build_half(nx,ny,forcing='neck-uniform'):
    assert forcing=='neck-uniform' and nx%4==0 and ny%16==0
    hx,hy=1/nx,.8/ny
    ix,iy=np.meshgrid(np.arange(nx),np.arange(ny//2),indexing='ij')
    active=((ix+.5)*hx>=.25)|((iy+.5)*hy>.35)
    ix,iy=ix[active],iy[active]
    cells=np.array([ix*(ny+1)+iy,ix*(ny+1)+iy+1,(ix+1)*(ny+1)+iy,(ix+1)*(ny+1)+iy+1]).T
    used=np.unique(cells);mapping=np.full((nx+1)*(ny+1),-1,dtype=int)
    mapping[used]=np.arange(len(used));cells=mapping[cells]
    x=(used//(ny+1))*hx;y=(used%(ny+1))*hy
    mx=hx/6*np.array([[2,1],[1,2]]);my=hy/6*np.array([[2,1],[1,2]])
    kx=np.array([[1,-1],[-1,1]])/hx;ky=np.array([[1,-1],[-1,1]])/hy
    km=np.kron(kx,my)+np.kron(mx,ky)
    rr=np.repeat(cells,4,axis=1).ravel();cc=np.tile(cells,(1,4)).ravel()
    K=coo_matrix((np.tile(km.ravel(),len(cells)),(rr,cc)),shape=(len(used),len(used))).tocsr()
    w=np.bincount(cells.ravel(),minlength=len(used))*hx*hy/4
    b=np.zeros(len(used))
    for j in range(round(.35/hy),ny//2):b[mapping[j:j+2]]+=hy/2
    lap=(diags(1/w)@K).tocsr();lap.sort_indices()
    assert abs(w.sum()-.3125)<1e-12 and abs(b.sum()-.05)<1e-13
    assert np.max(abs(K@np.ones(len(w))))<1e-10
    return x,y,cells,None,K,None,w,b,lap

def half_load(x,y,b):
    ids=np.flatnonzero(b>0);ids=ids[np.argsort(y[ids])]
    assert np.allclose(x[ids],0) and np.isclose(y[ids[0]],.35) and np.isclose(y[ids[-1]],.4)
    q,wq=np.polynomial.legendre.leggauss(8);result=np.zeros_like(b)
    for lo,hi in zip(ids[:-1],ids[1:]):
        h=y[hi]-y[lo];yy=(y[lo]+y[hi])/2+h*q/2
        g=np.sqrt(2)*np.cos(np.pi*(yy-.4)/.1)
        result[lo]+=h/2*np.sum(wq*g*(1-q)/2)
        result[hi]+=h/2*np.sum(wq*g*(1+q)/2)
    assert np.isclose(result.sum(),.1*np.sqrt(2)/np.pi,rtol=1e-12)
    return result

@njit(cache=True,parallel=True)
def stage(p,v,basep,basev,a,outp,outv,z,ip,indices,data,coeff,b,source,eta,beta,drive,dt):
    for i in prange(len(p)):
        z[i]=p[i]+eta*v[i]
    margin=1.
    for i in prange(len(p)):
        factor=1-2*beta*p[i];margin=min(margin,factor)
        k=ip[i];total=0.
        if ip[i+1]-k==9:
            left=indices[k+1];right=indices[k+7]
            total=coeff[0]*z[left-1]+coeff[1]*z[left]+coeff[2]*z[left+1]
            total+=coeff[3]*z[i-1]+coeff[4]*z[i]+coeff[5]*z[i+1]
            total+=coeff[6]*z[right-1]+coeff[7]*z[right]+coeff[8]*z[right+1]
        else:
            for j in range(k,ip[i+1]):total+=data[j]*z[indices[j]]
        a[i]=(-total+drive*source[i]-b[i]*v[i]+2*beta*v[i]**2)/factor
        outp[i]=basep[i]+dt*v[i];outv[i]=basev[i]+dt*a[i]
    return margin

@njit(cache=True,parallel=True)
def finish(p,v,p4,v4,v2,v3,a1,a2,a3,z,ip,indices,data,coeff,b,source,eta,beta,drive,dt):
    for i in prange(len(p)):z[i]=p4[i]+eta*v4[i]
    margin=1.
    for i in prange(len(p)):
        factor=1-2*beta*p4[i];margin=min(margin,factor)
        k=ip[i];total=0.
        if ip[i+1]-k==9:
            left=indices[k+1];right=indices[k+7]
            total=coeff[0]*z[left-1]+coeff[1]*z[left]+coeff[2]*z[left+1]
            total+=coeff[3]*z[i-1]+coeff[4]*z[i]+coeff[5]*z[i+1]
            total+=coeff[6]*z[right-1]+coeff[7]*z[right]+coeff[8]*z[right+1]
        else:
            for j in range(k,ip[i+1]):total+=data[j]*z[indices[j]]
        a4=(-total+drive*source[i]-b[i]*v4[i]+2*beta*v4[i]**2)/factor
        p[i]+=dt*(v[i]+2*v2[i]+2*v3[i]+v4[i])/6
        v[i]+=dt*(a1[i]+2*a2[i]+2*a3[i]+a4)/6
        margin=min(margin,1-2*beta*p[i])
        if not np.isfinite(p[i]) or not np.isfinite(v[i]):margin=min(margin,-np.inf)
    return margin

@njit(cache=True)
def cycle_fast(p,v,ip,indices,data,b,omega,beta,eta,amp,spp,cidx,sampling,pout,vout,aout,source):
    n=len(p);dt=2*np.pi/omega/spp;z=np.empty(n)
    a1=np.empty(n);a2=np.empty(n);a3=np.empty(n)
    p2=np.empty(n);v2=np.empty(n);p3=np.empty(n);v3=np.empty(n);p4=np.empty(n);v4=np.empty(n)
    first=0
    for i in range(n):
        if ip[i+1]-ip[i]==9:first=i;break
    coeff=data[ip[first]:ip[first+1]].copy()
    for i in range(n):
        k=ip[i]
        if ip[i+1]-k==9:
            for j in range(9):
                if abs(data[k+j]-coeff[j])>1e-10*max(abs(coeff[j]),1.):raise ValueError('Nonuniform stencil')
            if indices[k+4]!=i or indices[k+2]!=indices[k+1]+1 or indices[k]!=indices[k+1]-1 or indices[k+8]!=indices[k+7]+1 or indices[k+6]!=indices[k+7]-1:raise ValueError('Node ordering')
    row=0;margin=1.
    for step in range(spp):
        c=cidx+step/spp;ramp=.5*(1-np.cos(np.pi*min(c/30.,1.)))
        margin=min(margin,stage(p,v,p,v,a1,p2,v2,z,ip,indices,data,coeff,b,source,eta,beta,amp*ramp*np.sin(2*np.pi*c),dt/2))
        if row<len(sampling) and step==sampling[row]:
            pout[row]=p
            if vout.shape[0]>0:vout[row]=v;aout[row]=a1
            row+=1
        cm=c+.5/spp;ramp=.5*(1-np.cos(np.pi*min(cm/30.,1.)));drive=amp*ramp*np.sin(2*np.pi*cm)
        margin=min(margin,stage(p2,v2,p,v,a2,p3,v3,z,ip,indices,data,coeff,b,source,eta,beta,drive,dt/2))
        margin=min(margin,stage(p3,v3,p,v,a3,p4,v4,z,ip,indices,data,coeff,b,source,eta,beta,drive,dt))
        ce=c+1/spp;ramp=.5*(1-np.cos(np.pi*min(ce/30.,1.)))
        margin=min(margin,finish(p,v,p4,v4,v2,v3,a1,a2,a3,z,ip,indices,data,coeff,b,source,eta,beta,amp*ramp*np.sin(2*np.pi*ce),dt))
        if margin<=.02:raise ValueError('Invalid Westervelt state')
    return margin

@njit(cache=True,parallel=True)
def lifted_rhs(pb,pd,vb,vd,lb,ld,u,levels,src,rad,beta2):
    groups,n=pb.shape;nl=len(levels);out=np.empty((groups*nl,n),np.float32)
    mins=np.empty(groups*nl,np.float32);peaks=np.empty(groups*nl,np.float32)
    one=np.float32(1)
    for row in prange(groups*nl):
        group=row//nl;ell=levels[row%nl];small=one;large=np.float32(0)
        for j in range(n):
            p=pb[group,j]+ell*pd[group,j]
            vel=vb[group,j]+ell*vd[group,j]
            lap=lb[group,j]+ell*ld[group,j]
            mf=one-beta2*p
            out[row,j]=(-lap+u[group]*src[j]-rad[j]*vel+beta2*vel*vel)/mf
            small=min(small,mf);large=max(large,abs(p))
        mins[row]=small;peaks[row]=large
    return out,mins.min(),peaks.max()

def build(cfg):
    x,y,cells,_,stiffness,_,half_mass,boundary,lap=build_half(cfg["nx"],cfg["ny"])
    source=half_load(x,y,boundary)/half_mass
    ids=np.flatnonzero(np.isclose(x,0,atol=1e-14))
    ids=ids[np.argsort(y[ids])]
    radiation_mass=np.zeros_like(half_mass)
    h=np.diff(y[ids])
    radiation_mass[ids[:-1]]+=h/2
    radiation_mass[ids[1:]]+=h/2
    radiation=cfg["radiation_coefficient"]*radiation_mass/half_mass
    return x,y,half_mass,lap,source,radiation

def native_rhs(p,v,forcing,cfg,lap,source,radiation):
    factor=1-2*cfg["beta"]*p
    if factor.min()<=.02:
        raise ValueError("Invalid Westervelt effective mass")
    return (-lap@(p+cfg["eta"]*v)+forcing*source-radiation*v+2*cfg["beta"]*v*v)/factor

def drive(c,amp):
    return amp*.5*(1-np.cos(np.pi*min(c/30.,1.)))*np.sin(2*np.pi*c)
