"""Frozen supervised motion inference; never reads possession reference labels."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from referee_supervised import R,design
from decoder_experiment import ROOT,load
from pipeline.mall.common import CACHE
from pipeline.mall.layer4_possession_model import crossgame as X
import idsse_experiment as I
from pipeline.run import write_csv

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(output=None,verify=False):
    expected=json.loads((R/'deployment.json').read_text())['sha256'] if verify else None
    selected=json.loads((R/'frozen.json').read_text());game=load('tactical');z,F,L2,L3,l4=game
    mat,names,parts=design('tactical');modelpath=R/'final_firsthalves.pkl';prob=I.model(modelpath).predict_proba(mat)[:,1].astype('float32')
    # Use the current pipeline features, not probabilities from a prior experiment.
    a=np.load(Path(CACHE)/'models'/'design_tactical.npz');matrix=a['X'];feature_names=list(a['names']);fr=X.cp_candidates(np.zeros(len(prob)),z['half']);cx,_=X.cp_design(matrix,feature_names,L2,fr)
    cp_path=Path(CACHE)/'models'/'cp_bench1+bench2+bench3.pkl';cp=X.predict_cp(*I.model(cp_path),cx).astype('float32')
    p=dict(old=l4['p_home_full'],new=l4['p_home_full'],cp_frames=fr,cp_p=cp,dead=prob)
    pred=I.infer('tactical',selected['setting'],parts,p)
    path=Path(output) if output else (R/'replay.csv' if verify else ROOT.parent/'possession-viewer'/'data'/'mall-referee-supervised.csv')
    write_csv(pred,z['half'],path);digest=sha(path)
    if verify:
        assert digest==expected,'Prediction differs from frozen output'
        print('PASS byte-identical replay',digest,flush=True)
    else:
        np.save(R/'tactical_selected.npy',pred)
        manifest={'selection':selected,'path':str(path),'sha256':digest,'model_sha256':sha(modelpath),'cp_model_sha256':sha(cp_path),'motion_feature_names':names,'trained_on':'two external first halves only','radius_m':1.5,'decoder_switch_cost':4.,'hysteresis_strength':.30,'target_labels_used':False}
        (R/'deployment.json').write_text(json.dumps(manifest,indent=2));print('EXPORTED',str(path),digest,flush=True)
    return path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args();run(a.output,a.verify)
