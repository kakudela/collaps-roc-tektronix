#!/usr/bin/env python3
"""
RDataFrame analysis for one run of scope data (data/run_YYYYMMDD_HHMMSS/),
copied over from the DAQ laptop. Computes, per channel, per event:
  - baseline   (mean ADC value in the first N samples, before the pulse)
  - integral   (charge proxy: sum of baseline-minus-sample, converted to
                picocoulombs assuming 50-ohm termination)
  - peak       (pulse depth in mV, baseline minus the minimum sample)
  - peak_idx   (which sample the pulse peaks at -- used for timing)

A trigger firing on CH1 does not mean every outer PMT saw a real hit that
event -- most of the time only some of them did, the rest just show baseline
noise. So integral/peak/timing are all booked twice:
  "_all" -- every trigger, noise-dominated for channels with no real hit
  "_hit" -- only events where that channel's peak clears --hit-threshold-mv
The "_hit" versions are the physically meaningful ones.

Run on a machine with PyROOT (e.g. submit):
    python3 analyze_run.py /path/to/run_20260706_172716
"""
import argparse
import glob
import json
import os

import ROOT

import plot_utils

ROOT.gROOT.SetBatch(True)

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

DEFAULT_INDEX_PHP = os.path.expanduser("~/public_html/fccee/beam_background/index.php")


def main():
    ap = argparse.ArgumentParser(description="Analyze one run of MSO46 scope data with RDataFrame")
    ap.add_argument("run_dir", help="path to a run_YYYYMMDD_HHMMSS folder (rsynced from the DAQ laptop)")
    ap.add_argument("--baseline-samples", type=int, default=50,
                     help="how many samples at the start of the window to average for baseline")
    ap.add_argument("--termination-ohm", type=float, default=50.0)
    ap.add_argument("--hit-threshold-mv", type=float, default=15.0,
                     help="min peak depth (mV) for a channel to count as a real hit that event")
    ap.add_argument("--outdir", default=None, help="defaults to <run_dir>/analysis")
    ap.add_argument("--index-php-source", default=DEFAULT_INDEX_PHP,
                     help="index.php copied into every directory this script creates, for public_html browsing")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.threads > 0:
        ROOT.ROOT.EnableImplicitMT(args.threads)

    run_dir = args.run_dir.rstrip("/")

    meta = json.load(open(os.path.join(run_dir, "metadata.json")))
    trigger_ch = int(meta["trigger"]["source"].replace("CH", ""))
    channels = sorted(int(ch) for ch in meta["channels"].keys())
    outer = [c for c in channels if c != trigger_ch]
    xincr_ns = meta["channels"][str(trigger_ch)]["xincr"] * 1.0e9  # shared timebase, ns/sample

    files = sorted(glob.glob(os.path.join(run_dir, "batch*.root")))
    if not files:
        raise SystemExit(f"no batch*.root files found in {run_dir}")

    outdir = args.outdir or os.path.join(run_dir, "analysis")
    plot_utils.ensure_index_php(run_dir, args.index_php_source)
    plot_utils.ensure_index_php(outdir, args.index_php_source)

    df = ROOT.RDataFrame("events", files)
    n_events = df.Count().GetValue()
    print(f"Loaded {len(files)} files, {n_events} events, channels {channels}, trigger=CH{trigger_ch}")

    h1 = {}
    actions = []

    def book_h1(name, bins, col, node=None):
        src = node if node is not None else df
        h1[name] = src.Histo1D((name, "", bins[0], bins[1], bins[2]), col)
        actions.append(h1[name])

    thr = args.hit_threshold_mv

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
        book_h1(f"ch{ch}_peak_mv_all", (100, 0.0, 500.0), f"ch{ch}_peak_mv")
        book_h1(f"ch{ch}_integral_pC_all", (100, 0.0, 20.0), f"ch{ch}_integral_pC")

        if ch != trigger_ch:
            # "no real hit" events are mostly noise clustered near 0 -- filtering
            # them out is what actually makes the physical pulse population visible
            hit_node = df.Filter(f"ch{ch}_peak_mv > {thr}")
            book_h1(f"ch{ch}_peak_mv_hit", (100, 0.0, 500.0), f"ch{ch}_peak_mv", node=hit_node)
            book_h1(f"ch{ch}_integral_pC_hit", (100, 0.0, 20.0), f"ch{ch}_integral_pC", node=hit_node)

    # timing offset of each outer channel's peak relative to the trigger channel,
    # in real nanoseconds (not raw sample counts): "_all" = every trigger
    # (mostly noise for channels with no real hit), "_hit" = only events
    # clearing the hit threshold (the real coincidence timing)
    for ch in outer:
        df = df.Define(f"ch{ch}_dt_ns", f"(ch{ch}_peak_idx - ch{trigger_ch}_peak_idx) * {xincr_ns}")
        book_h1(f"ch{ch}_dt_ns_all", (100, -40.0, 40.0), f"ch{ch}_dt_ns")

        hit_node = df.Filter(f"ch{ch}_peak_mv > {thr}")
        book_h1(f"ch{ch}_dt_ns_hit", (100, -40.0, 40.0), f"ch{ch}_dt_ns", node=hit_node)

    ROOT.RDF.RunGraphs(actions)

    out_root = os.path.join(outdir, "analysis.root")
    tf = ROOT.TFile(out_root, "RECREATE")
    for name, h in h1.items():
        h.GetPtr().Write(name)
    tf.Close()
    print(f"Wrote histograms to {out_root}")

    # ── plots ──────────────────────────────────────────────────────────────
    for ch in channels:
        role = "trigger PMT" if ch == trigger_ch else "outer PMT"
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_integral_pC_all"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_integral_pC_all"),
            x_title=f"CH{ch} ({role}) charge integral [pC]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + " -- all triggers",
        )
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_peak_mv_all"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_peak_mv_all"),
            x_title=f"CH{ch} ({role}) pulse depth [mV]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + " -- all triggers",
        )
        if ch != trigger_ch:
            plot_utils.plot_hist_1d(
                h1[f"ch{ch}_integral_pC_hit"].GetPtr(),
                os.path.join(outdir, f"ch{ch}_integral_pC_hit"),
                x_title=f"CH{ch} ({role}) charge integral [pC]",
                y_title="events / bin",
                extra_left=plot_utils.HEADER_LEFT + f" -- peak > {thr:.0f}mV",
            )
            plot_utils.plot_hist_1d(
                h1[f"ch{ch}_peak_mv_hit"].GetPtr(),
                os.path.join(outdir, f"ch{ch}_peak_mv_hit"),
                x_title=f"CH{ch} ({role}) pulse depth [mV]",
                y_title="events / bin",
                extra_left=plot_utils.HEADER_LEFT + f" -- peak > {thr:.0f}mV",
            )

    # overlay of all outer-PMT integrals (hit-filtered), for a direct by-eye comparison
    plot_utils.plot_hists_1d(
        [h1[f"ch{ch}_integral_pC_hit"].GetPtr() for ch in outer],
        [f"CH{ch}" for ch in outer],
        os.path.join(outdir, "outer_pmts_integral_pC_overlay"),
        x_title="charge integral [pC]",
        y_title="events / bin",
        extra_left=plot_utils.HEADER_LEFT + f" -- peak > {thr:.0f}mV",
        canvas_size=(1000, 800),
    )

    for ch in outer:
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_dt_ns_all"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_dt_ns_all"),
            x_title=f"time of CH{ch} peak minus time of CH{trigger_ch} peak [ns]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + " -- all triggers",
        )
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_dt_ns_hit"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_dt_ns_hit"),
            x_title=f"time of CH{ch} peak minus time of CH{trigger_ch} peak [ns]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + f" -- peak > {thr:.0f}mV",
        )

    # CH2-5 timing overlays, both the "all triggers" and "hit-filtered" views,
    # on a bigger canvas so the 4-entry legend has room to breathe
    plot_utils.plot_hists_1d(
        [h1[f"ch{ch}_dt_ns_all"].GetPtr() for ch in outer],
        [f"CH{ch}" for ch in outer],
        os.path.join(outdir, "outer_pmts_dt_ns_all_overlay"),
        x_title=f"time of outer-PMT peak minus time of CH{trigger_ch} peak [ns]",
        y_title="events / bin",
        extra_left=plot_utils.HEADER_LEFT + " -- all triggers",
        canvas_size=(1000, 800),
    )
    plot_utils.plot_hists_1d(
        [h1[f"ch{ch}_dt_ns_hit"].GetPtr() for ch in outer],
        [f"CH{ch}" for ch in outer],
        os.path.join(outdir, "outer_pmts_dt_ns_hit_overlay"),
        x_title=f"time of outer-PMT peak minus time of CH{trigger_ch} peak [ns]",
        y_title="events / bin",
        extra_left=plot_utils.HEADER_LEFT + f" -- peak > {thr:.0f}mV",
        canvas_size=(1000, 800),
    )

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
