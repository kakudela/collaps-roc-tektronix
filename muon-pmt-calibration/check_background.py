#!/usr/bin/env python3
"""One-off check: of all CH1 triggers, how many show zero real hits on any
outer PMT? That population is the candidate for the unexplained ~1Hz
background -- CH1 fired, but nothing confirms a particle actually crossed
the outer scintillator too."""
import sys
import glob
import json

import ROOT

run_dir = sys.argv[1].rstrip("/")
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0

meta = json.load(open(f"{run_dir}/metadata.json"))
files = sorted(glob.glob(f"{run_dir}/batch*.root"))

ROOT.gInterpreter.Declare(r"""
double wb(const ROOT::VecOps::RVec<Short_t>& wf, int n){
    int m=std::min(n,(int)wf.size()); double s=0; for(int i=0;i<m;i++) s+=wf[i]; return s/m;
}
double wp(const ROOT::VecOps::RVec<Short_t>& wf, double b){
    double m=*std::min_element(wf.begin(), wf.end()); return b-m;
}
""")

df = ROOT.RDataFrame("events", files)
n_total = df.Count().GetValue()

hit_exprs = []
for ch in [2, 3, 4, 5]:
    ymult = meta["channels"][str(ch)]["ymult"]
    df = df.Define(f"b{ch}", f"wb(ch{ch}_raw,50)")
    df = df.Define(f"pk{ch}", f"wp(ch{ch}_raw,b{ch})*{ymult}*1000.0")
    df = df.Define(f"hit{ch}", f"pk{ch} > {thr}")
    hit_exprs.append(f"(int)hit{ch}")

df = df.Define("n_outer_hit", " + ".join(hit_exprs))

print(f"total triggers: {n_total}")
for n in range(5):
    cnt = df.Filter(f"n_outer_hit == {n}").Count().GetValue()
    print(f"  events with exactly {n} outer PMTs hit: {cnt}  ({100*cnt/n_total:.1f}%)")

n_zero = df.Filter("n_outer_hit == 0").Count().GetValue()
print()
print(f"CH1 fired but NO outer PMT confirmed a hit: {n_zero}/{n_total} = {100*n_zero/n_total:.1f}%")
