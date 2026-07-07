#!/usr/bin/env python3
"""
RDataFrame analysis for one run of scope data (data/run_YYYYMMDD_HHMMSS/),
copied over from the DAQ laptop. Computes, per channel, per event:
  - baseline   (mean ADC value in the first N samples, before the pulse)
  - integral   (charge proxy: sum of baseline-minus-sample, converted to
                picocoulombs assuming 50-ohm termination)
  - peak       (pulse depth in mV, baseline minus the minimum sample)
  - peak_idx   (which sample the pulse peaks at -- used for timing)

Run on a machine with PyROOT (e.g. submit):
    python3 analyze_run.py /path/to/run_20260706_172716 --outdir ~/public_html/.../run_20260706_172716
"""
import argparse
import glob
import json
import os

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(1110)

ROOT.gInterpreter.Declare(r"""
#include "ROOT/RVec.hxx"
#include <algorithm>
using namespace ROOT::VecOps;

double waveform_baseline(const RVec<Short_t>& wf, int n_baseline) {
    int n = std::min<int>(n_baseline, (int)wf.size());
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += wf[i];
    return sum / n;
}

// integral in raw ADC*sample units (baseline - sample, summed over the whole
// window); multiply by ymult (V/count) * xincr (s/sample) to get volt-seconds
double waveform_integral_raw(const RVec<Short_t>& wf, double baseline) {
    double sum = 0.0;
    for (auto v : wf) sum += (baseline - v);
    return sum;
}

double waveform_peak_raw(const RVec<Short_t>& wf, double baseline) {
    double minv = *std::min_element(wf.begin(), wf.end());
    return baseline - minv;   // positive raw-count pulse depth
}

int waveform_peak_index(const RVec<Short_t>& wf) {
    return (int)std::distance(wf.begin(), std::min_element(wf.begin(), wf.end()));
}
""")


def main():
    ap = argparse.ArgumentParser(description="Analyze one run of MSO46 scope data with RDataFrame")
    ap.add_argument("run_dir", help="path to a run_YYYYMMDD_HHMMSS folder (rsynced from the DAQ laptop)")
    ap.add_argument("--baseline-samples", type=int, default=50,
                     help="how many samples at the start of the window to average for baseline")
    ap.add_argument("--termination-ohm", type=float, default=50.0)
    ap.add_argument("--outdir", default=None, help="defaults to <run_dir>/analysis")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.threads > 0:
        ROOT.ROOT.EnableImplicitMT(args.threads)

    meta = json.load(open(os.path.join(args.run_dir, "metadata.json")))
    trigger_ch = int(meta["trigger"]["source"].replace("CH", ""))
    channels = sorted(int(ch) for ch in meta["channels"].keys())

    files = sorted(glob.glob(os.path.join(args.run_dir, "batch*.root")))
    if not files:
        raise SystemExit(f"no batch*.root files found in {args.run_dir}")

    outdir = args.outdir or os.path.join(args.run_dir, "analysis")
    os.makedirs(outdir, exist_ok=True)

    df = ROOT.RDataFrame("events", files)
    n_events = df.Count().GetValue()
    print(f"Loaded {len(files)} files, {n_events} events, channels {channels}, trigger=CH{trigger_ch}")

    h1 = {}
    h2 = {}
    actions = []

    def book_h1(name, bins, col):
        h1[name] = df.Histo1D((name, "", bins[0], bins[1], bins[2]), col)
        actions.append(h1[name])

    for ch in channels:
        cm = meta["channels"][str(ch)]
        ymult, xincr = cm["ymult"], cm["xincr"]
        R = args.termination_ohm

        df = df.Define(f"ch{ch}_baseline_raw", f"waveform_baseline(ch{ch}_raw, {args.baseline_samples})")
        df = df.Define(f"ch{ch}_peak_raw", f"waveform_peak_raw(ch{ch}_raw, ch{ch}_baseline_raw)")
        df = df.Define(f"ch{ch}_integral_raw", f"waveform_integral_raw(ch{ch}_raw, ch{ch}_baseline_raw)")
        df = df.Define(f"ch{ch}_peak_idx", f"waveform_peak_index(ch{ch}_raw)")

        # convert to physical units: mV for peak, picocoulombs for integral (Q = V*t/R)
        df = df.Define(f"ch{ch}_peak_mv", f"ch{ch}_peak_raw * {ymult} * 1000.0")
        df = df.Define(f"ch{ch}_integral_pC",
                        f"ch{ch}_integral_raw * {ymult} * {xincr} / {R} * 1.0e12")

        # NOTE: bin ranges below are rough starting guesses based on this
        # detector's earlier test data -- widen/rebin once you've looked at
        # the actual histograms for your real run.
        book_h1(f"ch{ch}_baseline_raw", (200, -2000, 2000), f"ch{ch}_baseline_raw")
        book_h1(f"ch{ch}_peak_mv", (200, 0.0, 500.0), f"ch{ch}_peak_mv")
        book_h1(f"ch{ch}_integral_pC", (200, 0.0, 20.0), f"ch{ch}_integral_pC")

    # timing offset of each channel's peak relative to the trigger channel
    for ch in channels:
        if ch == trigger_ch:
            continue
        df = df.Define(f"ch{ch}_dt_samples", f"ch{ch}_peak_idx - ch{trigger_ch}_peak_idx")
        book_h1(f"ch{ch}_dt_samples", (200, -500, 500), f"ch{ch}_dt_samples")

    # 2D correlations between outer-PMT integrals (do channels covering
    # overlapping geometry see correlated energy deposits?)
    outer = [c for c in channels if c != trigger_ch]
    for i in range(len(outer)):
        for j in range(i + 1, len(outer)):
            a, b = outer[i], outer[j]
            name = f"ch{a}_vs_ch{b}_integral_pC"
            h2[name] = df.Histo2D((name, "", 100, 0.0, 20.0, 100, 0.0, 20.0),
                                   f"ch{a}_integral_pC", f"ch{b}_integral_pC")
            actions.append(h2[name])

    ROOT.RDF.RunGraphs(actions)

    out_root = os.path.join(outdir, "analysis.root")
    tf = ROOT.TFile(out_root, "RECREATE")
    for name, h in {**h1, **h2}.items():
        h.GetPtr().Write(name)
    tf.Close()
    print(f"Wrote histograms to {out_root}")

    # quick PNGs for the plots you actually care about first
    for ch in channels:
        c = ROOT.TCanvas(f"c_integral_{ch}", "", 800, 600)
        h1[f"ch{ch}_integral_pC"].GetPtr().Draw()
        c.SaveAs(os.path.join(outdir, f"ch{ch}_integral_pC.png"))

    for ch in channels:
        if ch == trigger_ch:
            continue
        c = ROOT.TCanvas(f"c_dt_{ch}", "", 800, 600)
        h1[f"ch{ch}_dt_samples"].GetPtr().Draw()
        c.SaveAs(os.path.join(outdir, f"ch{ch}_dt_samples.png"))

    print(f"Wrote PNGs to {outdir}")


if __name__ == "__main__":
    main()
