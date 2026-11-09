"""Strict, reproducible submission serialization."""
import hashlib
import numpy as np
from . import config as c

def write(labels, path=c.OUTPUT):
    labels=np.asarray(labels,dtype='<U1').copy()
    assert labels.shape==(c.N,)
    graded=np.zeros(c.N,dtype=bool)
    for a,b in c.HALVES:
        graded[a:b]=True
    labels[~graded]='X'
    assert np.isin(labels[graded],['H','A','D']).all()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        handle.write('frame,label\n')
        handle.writelines(f'{i},{label}\n' for i,label in enumerate(labels))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    print(f'{path}: {c.N} rows, SHA-256 {digest}',flush=True)
    return digest
