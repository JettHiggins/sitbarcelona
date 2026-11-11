"""Referee-motion dead-ball model with real external labels, no ball inputs."""
import os,json,pickle,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from pipeline.mall.layer1_features.features import rolling_mean
from decoder_experiment import ROOT,OUT,load as target_load,quick
from deadball_experiment import PARAMS
import idsse_experiment as I
R=OUT/'referee_supervised';R.mkdir(exist_ok=True);IDS=I.TEST

def speed(mid):
    if mid=='tactical':
        a=np.load(ROOT.parent/'referee-model'/'referee_features.npz');n=len(target_load(mid)[0]['half']);s=np.full((n,3),np.nan)
        for lag in range(5):
            idx=a['frames']+lag;valid=idx<n;s[idx[valid]]=a['motion'][valid,:,0]
        return s
    a=np.load(I.D/mid/'officials.npz');xy=a['xy'];roles=list(a['roles']);order=[roles.index('referee')]+[i for i,r in enumerate(roles) if 'Assistant' in r]
    xy=xy[:,order];s=np.full((len(xy),3),np.nan);idx=np.arange(5,len(xy),5);v=np.minimum(np.linalg.norm(xy[idx]-xy[idx-5],axis=2)/.2,15)
    for lag in range(5):
        ii=idx+lag;valid=ii<len(xy);s[ii[valid]]=v[valid]
    return s

def design(mid,dropout=False):
    parts=target_load(mid)[:4] if mid=='tactical' else I.load(mid);z,F,L2,L3=parts;s=speed(mid)
    if dropout:
        template=np.isfinite(speed('tactical'));template=template[target_load('tactical')[0]['half']>0];rng=np.random.default_rng(22000+IDS.index(mid));mask=np.ones_like(s,dtype=bool)
        for a in range(0,len(s),7500):
            length=min(7500,len(s)-a);start=int(rng.integers(len(template)-length));mask[a:a+length]=template[start:start+length]
        s[~mask]=np.nan
    with np.errstate(all='ignore'):
        signals={'ref_main':s[:,0],'ref_assistant_min':np.fmin(s[:,1],s[:,2]),'ref_assistant_max':np.fmax(s[:,1],s[:,2]),'player_mean':F['mean_speed_all'],'player_team_min':np.fmin(F['mean_speed_h'],F['mean_speed_a']),'player_team_max':np.fmax(F['mean_speed_h'],F['mean_speed_a'])}
    cols=[];names=[]
    for k,v in signals.items():
        v=np.asarray(v,float)
        for w in [0,12,37,125]:cols.append(v if w==0 else rolling_mean(v,w));names.append(f'{k}@{w}')
        for cutoff in [.3,1.5,3.]:
            active=np.where(np.isfinite(v),(v<cutoff).astype(float),np.nan)
            for w in [37,125]:cols.append(rolling_mean(active,w));names.append(f'{k}_below{cutoff}@{w}')
    return np.column_stack(cols).astype('float32'),names,parts

def train(train_ids,name):
    path=R/(name+'.pkl')
    if path.exists():return I.model(path)
    mats=[];ys=[]
    for mid in train_ids:
        gt=np.load(I.D/mid/'labels.npy');half=np.load(I.D/mid/'layer0.npz')['half'];valid=(half==1)&(gt>=0)
        for dropout in [False,True]:
            a,names,_=design(mid,dropout);mats.append(a[valid][::5]);ys.append((gt[valid]==2)[::5])
    clf=HistGradientBoostingClassifier(**dict(PARAMS,max_iter=120,min_samples_leaf=100));clf.fit(np.vstack(mats),np.concatenate(ys));I.save_model(path,clf);print('REF TRAINED',name,train_ids,flush=True);return clf

def prob(mid,clf,cache):
    if cache.exists():return dict(np.load(cache))
    mat,names,parts=design(mid);p=I.probabilities(mid);p=dict(p);p['dead']=clf.predict_proba(mat)[:,1].astype('float32');np.savez_compressed(cache,**p);return p

def main():
    settings={'baseline':{}}
    for b in [.25,.5,1.]:
        for w in [.1,.5]:settings[f'ref_b{b}_w{w}']={'dead_blend':b,'dead_weight':w}
    rows={k:{'setting':s,'scores':{}} for k,s in settings.items()}
    for mid in IDS:
        clf=train([t for t in IDS if t!=mid],'loo_'+mid);parts=I.load(mid);p=prob(mid,clf,R/(mid+'_loo_prob.npz'));gt=np.load(I.D/mid/'labels.npy');target=np.where(parts[0]['half']==1,gt,3)
        for name,s in settings.items():
            pred=I.infer(mid,s,parts,p);rows[name]['scores'][mid]=quick(pred,target);print('REF VALID',mid,name,rows[name]['scores'][mid],flush=True)
        (R/'validation.json').write_text(json.dumps(rows,indent=2))
    base=rows['baseline']['scores'];eligible=[k for k,v in rows.items() if all(r['macro']>=base[t]['macro']-.003 for t,r in v['scores'].items())]
    best=max(eligible,key=lambda k:np.mean([r['spell'] for r in rows[k]['scores'].values()]));frozen={'name':best,**rows[best],'training':'first halves only; leave-one-match-out selection','test':'second halves; no second-half labels used in training or selection','features':'referee speed and player speed summaries only; no ball or target possession labels'};(R/'frozen.json').write_text(json.dumps(frozen,indent=2));print('REF FROZEN',best,flush=True)
    clf=train(IDS,'final_firsthalves');report={}
    for mid in IDS:
        parts=I.load(mid);p=prob(mid,clf,R/(mid+'_test_prob.npz'));gt=np.load(I.D/mid/'labels.npy');target=np.where(parts[0]['half']==2,gt,3);report[mid]={}
        for name,s in [('baseline',{}),('selected',frozen['setting'])]:
            pred=I.infer(mid,s,parts,p);np.save(R/(mid+'_'+name+'.npy'),pred);report[mid][name]=quick(pred,target);print('REF TEST',mid,name,report[mid][name],flush=True)
        (R/'test.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
