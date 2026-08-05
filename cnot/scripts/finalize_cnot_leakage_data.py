#!/usr/bin/env python3
"""Combine optimized CNOT pulses and leakage-inclusive noisy logs."""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd


def last_array(path, name):
    text = Path(path).read_text(encoding='utf-8', errors='replace')
    blocks = re.findall(rf'{re.escape(name)}\s*=\s*np\.array\(\[(.*?)\]\)', text, re.S)
    if not blocks:
        raise ValueError(f'{name!r} not found in {path}')
    return np.asarray([float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', blocks[-1])])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--optimization-csv',required=True)
    ap.add_argument('--noise-170',required=True)
    ap.add_argument('--noise-30',required=True)
    ap.add_argument('--noise-3',required=True)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()
    opt=pd.read_csv(a.optimization_csv)
    out=pd.DataFrame({
      'tg':opt.p0,'drive_amp_1':opt.p1,'drive_amp_2':opt.p2,
      'detune_1':opt.p3,'detune_2':opt.p4,
      'f_leakage_220':opt.f_trunc,
      'fidelity_leakage_220':1-10**opt.f_trunc,
      'f_leakage_1000':opt.f_large,
      'fidelity_leakage_1000':1-10**opt.f_large,
    })
    for t1,path in [(170,a.noise_170),(30,a.noise_30),(3,a.noise_3)]:
      vals=last_array(path,'fidelity_noise_list')
      if len(vals)!=len(out): raise ValueError(f'T1={t1}: {len(vals)} results for {len(out)} pulses')
      out[f'f_leakage_220_{t1}us']=vals
      out[f'fidelity_leakage_220_{t1}us']=1-10**vals
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True)
    # Pulse parameters use six decimals; every fidelity-related number uses eight.
    lines=[','.join(out.columns)]
    for _,row in out.iterrows():
      fields=[]
      for c in out.columns:
        fields.append(f'{row[c]:.8f}' if c.startswith(('f_leakage_','fidelity_leakage_')) else f'{row[c]:.6f}')
      lines.append(','.join(fields))
    p.write_text('\n'.join(lines)+'\n')
    print(f'wrote {len(out)} rows to {p}')

if __name__=='__main__': main()
