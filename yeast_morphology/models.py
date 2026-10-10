"""Train-only target representation, ridge, Gaussian MLP, and conditional DDPM."""
from __future__ import annotations
import copy, math, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

class Representation:
    def fit(self, x, y, latent_dim=32):
        self.keep=(np.isfinite(y).mean(axis=0)>=.95) & (np.nanstd(y,axis=0)>1e-8)
        if self.keep.sum()<2: raise ValueError('Too few usable training traits')
        yy=y[:,self.keep]
        self.median=np.nanmedian(yy,axis=0)
        yy=np.where(np.isfinite(yy),yy,self.median)
        self.xscale=StandardScaler().fit(x)
        self.yscale=StandardScaler().fit(yy)
        dim=min(latent_dim,len(y)-1,int(self.keep.sum()))
        self.pca=PCA(n_components=dim,whiten=True,svd_solver='full').fit(self.yscale.transform(yy))
        return self
    def x(self,x): return self.xscale.transform(x).astype('float32')
    def y(self,y):
        yy=y[:,self.keep]; yy=np.where(np.isfinite(yy),yy,self.median)
        return self.pca.transform(self.yscale.transform(yy)).astype('float32')
    def decode(self,z):
        shape=z.shape[:-1]
        out=self.yscale.inverse_transform(self.pca.inverse_transform(z.reshape(-1,z.shape[-1])))
        return out.reshape(*shape,-1).astype('float32')
    def standardized(self,y): return (y[:,self.keep]-self.yscale.mean_)/self.yscale.scale_

class GaussianNet(nn.Module):
    def __init__(self,xdim,zdim,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(xdim,width),nn.SiLU(),nn.Dropout(.1),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2*zdim))
    def forward(self,x):
        mu,lv=self.net(x).chunk(2,-1)
        return mu,lv.clamp(-5,3)

class Denoiser(nn.Module):
    def __init__(self,xdim,zdim,width=128):
        super().__init__()
        self.cond=nn.Sequential(nn.Linear(xdim,width),nn.SiLU(),nn.Linear(width,64))
        self.net=nn.Sequential(nn.Linear(zdim+64+32,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,zdim))
    def forward(self,z,t,x):
        freq=torch.exp(torch.arange(16,device=z.device)*(-math.log(10000)/15))
        angles=t[:,None]*freq[None,:]
        return self.net(torch.cat([z,self.cond(x),angles.sin(),angles.cos()],-1))

def schedule(steps,device):
    # Cosine schedule from Nichol & Dhariwal, with standard discrete beta cap.
    t=torch.linspace(0,steps,steps+1,device=device)/steps
    a=torch.cos((t+.008)/1.008*math.pi/2).square(); a=a/a[0]
    beta=(1-a[1:]/a[:-1]).clamp(.0001,.999)
    return beta,torch.cumprod(1-beta,0)

def device_for(name='auto'):
    if name=='auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA requested but unavailable')
    return name

def fit_neural(kind,x,z,xval,zval,epochs=120,seed=20261010,device='cpu',steps=100,width=128,max_seconds=600):
    torch.manual_seed(seed); np.random.seed(seed)
    device=device_for(device)
    model=(GaussianNet if kind=='gaussian' else Denoiser)(x.shape[1],z.shape[1],width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    xx=torch.tensor(x,device=device); zz=torch.tensor(z,device=device)
    xv=torch.tensor(xval,device=device); zv=torch.tensor(zval,device=device)
    beta,abar=schedule(steps,device)
    # Identical noise and timesteps at every validation checkpoint.
    gen=torch.Generator(device=device).manual_seed(seed+17)
    vt=torch.randint(steps,(len(xv),),device=device,generator=gen)
    ve=torch.randn(zv.shape,device=device,generator=gen)
    best=float('inf'); state=None; history=[]; bad=0; start=time.monotonic(); stopped='epochs'
    def loss(xb,zb,validation=False):
        if kind=='gaussian':
            mu,lv=model(xb); return .5*(lv+(zb-mu).square()*torch.exp(-lv)).mean()
        t=vt if validation else torch.randint(steps,(len(xb),),device=device)
        noise=ve if validation else torch.randn_like(zb)
        at=abar[t,None]
        noisy=at.sqrt()*zb+(1-at).sqrt()*noise
        # Velocity prediction avoids division by tiny terminal signal-to-noise ratios.
        target=at.sqrt()*noise-(1-at).sqrt()*zb
        return (model(noisy,t.float(),xb)-target).square().mean()
    for epoch in range(epochs):
        model.train(); total=0
        order=torch.randperm(len(x),device=device)
        for ix in order.split(128):
            opt.zero_grad(set_to_none=True); value=loss(xx[ix],zz[ix])
            if not torch.isfinite(value): raise RuntimeError(f'{kind} training diverged')
            value.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); total+=value.item()*len(ix)
        model.eval()
        with torch.no_grad(): val=float(loss(xv,zv,True))
        history.append({'epoch':epoch+1,'train_loss':total/len(x),'val_loss':val})
        if val<best-1e-5:
            best=val; state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; bad=0
        else: bad+=1
        if (epoch+1)%20==0: print(f'{kind}: epoch {epoch+1}, val={val:.4f}',flush=True)
        if bad>=25: stopped='early_stopping'; break
        if time.monotonic()-start>=max_seconds: stopped='time_limit'; break
    model.load_state_dict(state); model.eval()
    return {'kind':kind,'state':{k:v.cpu() for k,v in model.state_dict().items()},
        'parameterization':'velocity','xdim':x.shape[1],'zdim':z.shape[1],'width':width,'steps':steps,'seed':seed,
        'best_epoch':best_epoch,'best_val_loss':best,'stopped':stopped,'history':history,'seconds':time.monotonic()-start}

def neural_samples(checkpoint,x,n=32,seed=20261010,device='cpu'):
    device=device_for(device); kind=checkpoint['kind']
    model=(GaussianNet if kind=='gaussian' else Denoiser)(checkpoint['xdim'],checkpoint['zdim'],checkpoint['width']).to(device)
    model.load_state_dict(checkpoint['state']); model.eval()
    gen=torch.Generator(device=device).manual_seed(seed)
    out=[]; steps=checkpoint['steps']; beta,abar=schedule(steps,device)
    with torch.no_grad():
        for begin in range(0,len(x),128):
            xx=torch.tensor(x[begin:begin+128],device=device).repeat_interleave(n,0)
            noise=torch.randn((len(xx),checkpoint['zdim']),device=device,generator=gen)
            if kind=='gaussian':
                mu,lv=model(xx); zz=mu+torch.exp(.5*lv)*noise
            else:
                zz=noise
                for t in reversed(range(steps)):
                    tt=torch.full((len(xx),),float(t),device=device)
                    pred=model(zz,tt,xx)
                    if checkpoint.get('parameterization')=='velocity':
                        x0=abar[t].sqrt()*zz-(1-abar[t]).sqrt()*pred
                        prev=abar[t-1] if t else torch.tensor(1.,device=device)
                        zz=prev.sqrt()*beta[t]/(1-abar[t])*x0+(1-beta[t]).sqrt()*(1-prev)/(1-abar[t])*zz
                    else:  # Read legacy smoke checkpoints, never silently reinterpret them.
                        zz=(zz-beta[t]/(1-abar[t]).sqrt()*pred)/(1-beta[t]).sqrt()
                    if t:
                        var=beta[t]*(1-abar[t-1])/(1-abar[t])
                        zz+=var.sqrt()*torch.randn(zz.shape,device=device,generator=gen)
            out.append(zz.reshape(-1,n,checkpoint['zdim']).cpu().numpy())
    result=np.concatenate(out)
    if not np.isfinite(result).all(): raise RuntimeError('Non-finite neural samples')
    return result

def fit_ridge(x,z,xval,zval):
    candidates=[]
    for alpha in [1.,10.,100.,1000.,10000.]:
        model=Ridge(alpha=alpha).fit(x,z)
        candidates.append((float(np.mean((model.predict(xval)-zval)**2)),alpha,model))
    score,alpha,model=min(candidates,key=lambda t:t[0])
    residual=zval-model.predict(xval)
    residual-=residual.mean(axis=0)
    return model,residual,{'alpha':alpha,'val_mse_latent':score}

def ridge_samples(model,residual,x,n,seed):
    rng=np.random.default_rng(seed)
    return model.predict(x)[:,None,:]+residual[rng.integers(len(residual),size=(len(x),n))]
