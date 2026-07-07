#!/usr/bin/env python3
"""
RDataFrame analysis for one run of scope data (data/run_YYYYMMDD_HHMMSS/),
copied over from the DAQ laptop. Computes, per channel, per event:
  - baseline   (mean ADC value in the first N samples, before the pulse)
  - integral   (charge proxy: sum of baseline-minus-sample, converted to
                picocoulombs assuming 50-ohm termination)
  - peak       (pulse depth in mV, baseline minus the minimum sample)
  - peak_idx   (which sample the pulse peaks at, used for timing)
  - n_outer_hit (0-4: how many of the outer PMTs registered a real hit)

A trigger firing on CH1 does not mean every outer PMT saw a real hit that
event, most of the time only some of them did, the rest just show baseline
noise. So integral/peak/timing are all booked twice:
  "_all" - every trigger, noise-dominated for channels with no real hit
  "_hit" - only events where that channel's peak clears --hit-threshold-mv
The "_hit" versions are the physically meaningful ones.

Every run also prints (and saves to summary.txt) a full numeric report:
metadata, trigger rate, per-channel hit fractions, outer-PMT multiplicity
breakdown, timing offsets, and Landau fit results. The report contains
numbers and results only, no interpretation of what they mean.

Run on a machine with PyROOT (e.g. submit):
    python3 analyze_run.py /path/to/run_20260706_172716
"""
import argparse
import glob
import json
import os
from datetime import datetime

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


def fit_landau(hist, fit_lo, fit_hi, rebin_factor=1):
    """Fit a Landau distribution (the standard shape for charged-particle
    energy loss through a fixed thickness of material) to a charge-integral
    histogram. Returns (params_dict_or_None, TF1).

    fit_lo is set above 0 deliberately: events near the hit-threshold cut are
    partly shaped by that selection, not pure physics, so including the very
    bottom of the distribution would bias the fit. fit_hi excludes the
    sparsest part of the far tail, where per-bin statistics are too low to
    usefully constrain the fit.

    rebin_factor merges this many display-bins into one wider "fit bin"
    before fitting. The fit is genuinely sensitive to bin width: chi-squared
    per bin uses sqrt(counts) as the expected statistical noise, so if bins
    are so fine that some only have a handful of entries (or zero), that
    noise estimate gets noisy/unreliable itself, which can inflate chi2/ndf
    for reasons that have nothing to do with whether the Landau shape is
    actually a good description of the data. Rebinning for the fit only
    (the displayed histogram is untouched) keeps the fit on bins with
    healthy statistics regardless of how finely the plot itself is binned."""
    hfit = hist.Clone(f"{hist.GetName()}_fitcopy")
    hfit.SetDirectory(0)
    if rebin_factor > 1:
        hfit.Rebin(rebin_factor)

    name = f"landau_{hist.GetName()}"
    f = ROOT.TF1(name, "landau", fit_lo, fit_hi)
    peak_x = hfit.GetXaxis().GetBinCenter(hfit.GetMaximumBin())
    f.SetParameters(hfit.GetMaximum(), peak_x, max(hfit.GetRMS() * 0.5, 0.5))
    fit_result = hfit.Fit(f, "SQR")  # S=return result, Q=quiet, R=use f's own range
    if int(fit_result) != 0:
        return None, f
    ndf = f.GetNDF()
    params = dict(
        mpv=f.GetParameter(1), mpv_err=f.GetParError(1),
        sigma=f.GetParameter(2), sigma_err=f.GetParError(2),
        chi2_ndf=(f.GetChisquare() / ndf) if ndf > 0 else float("nan"),
    )

    # The fit's height (param 0) was determined against hfit's wider bins,
    # which each contain ~rebin_factor times more events than the original
    # display histogram's bins. Drawing that curve as-is on top of the fine
    # display histogram would overshoot by that same factor, so rescale it
    # back down so the curve's height actually matches the bars it's drawn
    # over. MPV and sigma (params 1, 2) describe the curve's x-axis shape,
    # not its height, so they're untouched.
    if rebin_factor > 1:
        f.SetParameter(0, f.GetParameter(0) / rebin_factor)

    # attach the fit to the actual (fine-binned) histogram too, purely so it
    # gets saved to the output ROOT file alongside it
    hist.GetListOfFunctions().Add(f)
    return params, f


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
    run_label = os.path.basename(run_dir)

    report = []

    def log(line=""):
        print(line)
        report.append(line)

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

    log("=" * 72)
    log("COLLAPS ROC Muon DAQ - analysis report")
    log(f"Run: {run_label}")
    log(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    log("=" * 72)
    log()
    log("- Scope settings (from metadata.json)")
    log(f"Instrument: {meta.get('idn', '?')}")
    log(f"Sample rate: {meta.get('sample_rate_hz', 0) / 1e9:.2f} GS/s")
    log(f"Record length: {meta.get('record_length', '?')} samples")
    trig = meta.get("trigger", {})
    log(f"Trigger: {trig.get('source')}, {trig.get('slope')} edge, "
        f"level = {trig.get('level_v', 0) * 1000:.1f} mV, mode = {trig.get('mode')}")
    for ch in channels:
        cm = meta["channels"][str(ch)]
        role = "trigger PMT" if ch == trigger_ch else "outer PMT"
        log(f"  CH{ch} ({role}): {cm['scale_v_div'] * 1000:.0f} mV/div, "
            f"{cm['termination_ohm']:.0f} ohm, BW = {cm['bandwidth_hz'] / 1e9:.2f} GHz, "
            f"coupling = {cm['coupling']}")
    log()
    log(f"Analysis settings: hit threshold = {args.hit_threshold_mv:.0f} mV peak, "
        f"baseline = first {args.baseline_samples} samples, termination = {args.termination_ohm:.0f} ohm")
    log()

    df = ROOT.RDataFrame("events", files)
    n_events = df.Count().GetValue()
    log(f"Files: {len(files)} batch files, {n_events} total events, "
        f"channels {channels}, trigger = CH{trigger_ch}")
    log()

    h1 = {}
    actions = []
    plot_descriptions = []

    def book_h1(name, bins, col, node=None):
        src = node if node is not None else df
        h1[name] = src.Histo1D((name, "", bins[0], bins[1], bins[2]), col)
        actions.append(h1[name])

    def add_plot(name, desc):
        plot_descriptions.append((name, desc))

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
        # detector's earlier test data, widen/rebin once you've looked at
        # the actual histograms for your real run.
        #
        # integral_pC range is 0-60pC: 0-20pC silently cut off 18-33% of
        # events per channel (worst for CH5), the real tail extends out to
        # 100-250pC in rare high-energy events. 0-60pC captures 97-99%.
        #
        # Display binning is much finer than the fit binning: fine bins
        # make the plot show more real structure, but fitting a Landau
        # directly against bins this fine would mean very few events per
        # bin, which destabilizes the fit. fit_landau() works off a coarser
        # rebinned copy internally, decoupled from the display binning.
        book_h1(f"ch{ch}_peak_mv_all", (500, 0.0, 500.0), f"ch{ch}_peak_mv")
        book_h1(f"ch{ch}_integral_pC_all", (120, 0.0, 60.0), f"ch{ch}_integral_pC")

        if ch != trigger_ch:
            df = df.Define(f"ch{ch}_is_hit", f"ch{ch}_peak_mv > {thr}")
            hit_node = df.Filter(f"ch{ch}_is_hit")
            book_h1(f"ch{ch}_peak_mv_hit", (500, 0.0, 500.0), f"ch{ch}_peak_mv", node=hit_node)
            book_h1(f"ch{ch}_integral_pC_hit", (120, 0.0, 60.0), f"ch{ch}_integral_pC", node=hit_node)

    # timing offset of each outer channel's peak relative to the trigger
    # channel, in real nanoseconds (not raw sample counts)
    for ch in outer:
        df = df.Define(f"ch{ch}_dt_ns", f"(ch{ch}_peak_idx - ch{trigger_ch}_peak_idx) * {xincr_ns}")
        book_h1(f"ch{ch}_dt_ns_all", (400, -40.0, 40.0), f"ch{ch}_dt_ns")

        hit_node = df.Filter(f"ch{ch}_is_hit")
        book_h1(f"ch{ch}_dt_ns_hit", (400, -40.0, 40.0), f"ch{ch}_dt_ns", node=hit_node)

    # multiplicity: how many of the 4 outer PMTs registered a real hit this event
    df = df.Define("n_outer_hit", " + ".join(f"(int)ch{ch}_is_hit" for ch in outer))
    book_h1("n_outer_hit", (5, -0.5, 4.5), "n_outer_hit")

    # run duration, from the (repeated-per-event) batch timing columns
    df = df.Define("batch_end_unix", "batch_start_unix + duration_s")
    t_start_action = df.Min("batch_start_unix")
    t_end_action = df.Max("batch_end_unix")
    actions += [t_start_action, t_end_action]

    ROOT.RDF.RunGraphs(actions)
    duration_s = t_end_action.GetValue() - t_start_action.GetValue()

    # plots
    landau_fits = {}
    for ch in channels:
        role = "trigger PMT" if ch == trigger_ch else "outer PMT"
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_integral_pC_all"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_integral_pC_all"),
            x_title=f"CH{ch} ({role}) charge integral [pC]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + " (all triggers)",
        )
        add_plot(f"ch{ch}_integral_pC_all", f"CH{ch} charge integral [pC], all triggers")
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_peak_mv_all"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_peak_mv_all"),
            x_title=f"CH{ch} ({role}) pulse depth [mV]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + " (all triggers)",
        )
        add_plot(f"ch{ch}_peak_mv_all", f"CH{ch} pulse depth [mV], all triggers")
        if ch != trigger_ch:
            # integral_pC_hit is booked at 0.5pC/bin (120 bins over 60pC) for
            # a fine-grained plot; rebin_factor=2 merges that back to 1pC/bin
            # for the fit itself, which is the bin width already confirmed
            # to give stable chi2/ndf around 1 for this amount of statistics.
            fit_params, fit_func = fit_landau(h1[f"ch{ch}_integral_pC_hit"].GetPtr(),
                                               fit_lo=2.0, fit_hi=40.0, rebin_factor=2)
            landau_fits[ch] = fit_params
            if fit_params:
                annotation = (f"Landau fit: MPV = {fit_params['mpv']:.2f} #pm {fit_params['mpv_err']:.2f} pC\n"
                              f"#chi^{{2}}/ndf = {fit_params['chi2_ndf']:.2f}")
            else:
                annotation = "Landau fit did not converge"
            plot_utils.plot_hist_1d(
                h1[f"ch{ch}_integral_pC_hit"].GetPtr(),
                os.path.join(outdir, f"ch{ch}_integral_pC_hit"),
                x_title=f"CH{ch} ({role}) charge integral [pC]",
                y_title="events / bin",
                extra_left=plot_utils.HEADER_LEFT + f" (peak > {thr:.0f}mV)",
                fit_func=fit_func,
                annotation=annotation,
            )
            if fit_params:
                add_plot(f"ch{ch}_integral_pC_hit",
                         f"CH{ch} charge integral [pC], peak > {thr:.0f}mV. "
                         f"Landau fit: MPV = {fit_params['mpv']:.2f} +/- {fit_params['mpv_err']:.2f} pC, "
                         f"sigma = {fit_params['sigma']:.2f} +/- {fit_params['sigma_err']:.2f} pC, "
                         f"chi2/ndf = {fit_params['chi2_ndf']:.2f}")
            else:
                add_plot(f"ch{ch}_integral_pC_hit",
                         f"CH{ch} charge integral [pC], peak > {thr:.0f}mV. Landau fit did not converge")
            plot_utils.plot_hist_1d(
                h1[f"ch{ch}_peak_mv_hit"].GetPtr(),
                os.path.join(outdir, f"ch{ch}_peak_mv_hit"),
                x_title=f"CH{ch} ({role}) pulse depth [mV]",
                y_title="events / bin",
                extra_left=plot_utils.HEADER_LEFT + f" (peak > {thr:.0f}mV)",
            )
            add_plot(f"ch{ch}_peak_mv_hit", f"CH{ch} pulse depth [mV], peak > {thr:.0f}mV")

    # overlay of all outer-PMT integrals, hit-filtered
    plot_utils.plot_hists_1d(
        [h1[f"ch{ch}_integral_pC_hit"].GetPtr() for ch in outer],
        [f"CH{ch}" for ch in outer],
        os.path.join(outdir, "outer_pmts_integral_pC_overlay"),
        x_title="charge integral [pC]",
        y_title="events / bin",
        extra_left=plot_utils.HEADER_LEFT + f" (peak > {thr:.0f}mV)",
        canvas_size=(1000, 800),
    )
    add_plot("outer_pmts_integral_pC_overlay", f"CH2-5 charge integral [pC] overlaid, peak > {thr:.0f}mV")

    for ch in outer:
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_dt_ns_all"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_dt_ns_all"),
            x_title=f"time of CH{ch} peak minus time of CH{trigger_ch} peak [ns]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + " (all triggers)",
        )
        add_plot(f"ch{ch}_dt_ns_all", f"CH{ch} timing relative to CH{trigger_ch} [ns], all triggers")
        plot_utils.plot_hist_1d(
            h1[f"ch{ch}_dt_ns_hit"].GetPtr(),
            os.path.join(outdir, f"ch{ch}_dt_ns_hit"),
            x_title=f"time of CH{ch} peak minus time of CH{trigger_ch} peak [ns]",
            y_title="events / bin",
            extra_left=plot_utils.HEADER_LEFT + f" (peak > {thr:.0f}mV)",
        )
        h_dt = h1[f"ch{ch}_dt_ns_hit"].GetPtr()
        add_plot(f"ch{ch}_dt_ns_hit",
                 f"CH{ch} timing relative to CH{trigger_ch} [ns], peak > {thr:.0f}mV. "
                 f"mean = {h_dt.GetMean():.2f} ns, RMS = {h_dt.GetRMS():.2f} ns")

    # CH2-5 timing overlays, both the "all triggers" and "hit-filtered" views,
    # on a bigger canvas so the 4-entry legend has room to breathe
    plot_utils.plot_hists_1d(
        [h1[f"ch{ch}_dt_ns_all"].GetPtr() for ch in outer],
        [f"CH{ch}" for ch in outer],
        os.path.join(outdir, "outer_pmts_dt_ns_all_overlay"),
        x_title=f"time of outer-PMT peak minus time of CH{trigger_ch} peak [ns]",
        y_title="events / bin",
        extra_left=plot_utils.HEADER_LEFT + " (all triggers)",
        canvas_size=(1000, 800),
    )
    add_plot("outer_pmts_dt_ns_all_overlay", f"CH2-5 timing relative to CH{trigger_ch} [ns] overlaid, all triggers")
    plot_utils.plot_hists_1d(
        [h1[f"ch{ch}_dt_ns_hit"].GetPtr() for ch in outer],
        [f"CH{ch}" for ch in outer],
        os.path.join(outdir, "outer_pmts_dt_ns_hit_overlay"),
        x_title=f"time of outer-PMT peak minus time of CH{trigger_ch} peak [ns]",
        y_title="events / bin",
        extra_left=plot_utils.HEADER_LEFT + f" (peak > {thr:.0f}mV)",
        canvas_size=(1000, 800),
    )
    add_plot("outer_pmts_dt_ns_hit_overlay",
             f"CH2-5 timing relative to CH{trigger_ch} [ns] overlaid, peak > {thr:.0f}mV")

    # outer-PMT hit multiplicity
    plot_utils.plot_hist_1d(
        h1["n_outer_hit"].GetPtr(),
        os.path.join(outdir, "n_outer_hit_multiplicity"),
        x_title="number of outer PMTs with a real hit (0-4)",
        y_title="events",
        extra_left=plot_utils.HEADER_LEFT + f" (peak > {thr:.0f}mV)",
    )
    add_plot("n_outer_hit_multiplicity",
             f"Number of outer PMTs (CH2-5) with peak > {thr:.0f}mV, per event (0-4)")

    print(f"Wrote plots to {outdir}")

    # Written after fitting (not before) so the fitted Landau curves are
    # attached to their histograms and saved.
    out_root = os.path.join(outdir, "analysis.root")
    tf = ROOT.TFile(out_root, "RECREATE")
    for name, h in h1.items():
        h.GetPtr().Write(name)
    tf.Close()
    print(f"Wrote histograms (with fit results attached) to {out_root}")

    # report: trigger rate, hit fractions, multiplicity, timing, fits
    log()
    log("- Trigger rate")
    log(f"Run duration: {duration_s:.1f} s ({duration_s / 60:.1f} min)")
    log(f"Overall trigger rate (CH{trigger_ch}): {n_events / duration_s:.2f} Hz")
    log()

    log(f"- Outer-PMT hit fractions (peak > {thr:.0f}mV)")
    for ch in outer:
        n_hit = h1[f"ch{ch}_integral_pC_hit"].GetPtr().GetEntries()
        log(f"CH{ch}: {int(n_hit)}/{n_events} = {100 * n_hit / n_events:.1f}%  ({n_hit / duration_s:.2f} Hz)")
    log()

    log("- Outer-PMT hit multiplicity (how many of CH2-5 fired together)")
    hmult = h1["n_outer_hit"].GetPtr()
    for n in range(5):
        c = hmult.GetBinContent(n + 1)
        log(f"  {n} outer PMTs hit: {int(c):5d}  ({100 * c / n_events:5.1f}%,  {c / duration_s:.2f} Hz)")
    log()

    log(f"- Timing offsets (peak > {thr:.0f}mV)")
    for ch in outer:
        h = h1[f"ch{ch}_dt_ns_hit"].GetPtr()
        log(f"CH{ch}: mean dt = {h.GetMean():6.2f} ns   RMS = {h.GetRMS():.2f} ns   (n = {int(h.GetEntries())})")
    log()

    log("- Landau fit: charge-integral MPV per channel")
    for ch in outer:
        p = landau_fits.get(ch)
        if p:
            log(f"CH{ch}: MPV = {p['mpv']:.2f} +/- {p['mpv_err']:.2f} pC   "
                f"sigma = {p['sigma']:.2f} +/- {p['sigma_err']:.2f} pC   "
                f"chi2/ndf = {p['chi2_ndf']:.2f}")
        else:
            log(f"CH{ch}: fit did not converge")
    log()

    log("- Plots produced (each as .png and .pdf)")
    for name, desc in plot_descriptions:
        log(f"{name}: {desc}")
    log()
    log(f"Full raw scope settings: {os.path.join(run_dir, 'metadata.json')}")
    log(f"All histograms + fit objects: {out_root}")

    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"\nWrote full report to {summary_path}")


if __name__ == "__main__":
    main()
