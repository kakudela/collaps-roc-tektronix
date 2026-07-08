"""
Lightweight ROOT plotting helpers for the COLLAPS ROC muon DAQ.

Same header/margin style as the FCC-ee plot_utils.py, just stripped down
to plain 1D/2D histograms since this project doesn't need the acceptance
overlays or step-graph stuff.
"""
import os
import shutil

import ROOT

HEADER_LEFT = "#bf{COLLAPS ROC} #scale[0.7]{#it{Muon DAQ}}"


def ensure_index_php(dirpath, index_php_source):
    """Copy index.php into dirpath if it's not already there. Every dir
    under public_html needs its own copy for the browsing to work right."""
    os.makedirs(dirpath, exist_ok=True)
    dest = os.path.join(dirpath, "index.php")
    if not os.path.exists(dest) and index_php_source and os.path.exists(index_php_source):
        shutil.copy(index_php_source, dest)


def _style_axes(h):
    h.GetXaxis().SetTitleSize(0.045)
    h.GetYaxis().SetTitleSize(0.045)
    h.GetXaxis().SetLabelSize(0.04)
    h.GetYaxis().SetLabelSize(0.04)
    h.GetXaxis().SetTitleOffset(1.15)
    h.GetYaxis().SetTitleOffset(1.5)
    h.GetXaxis().SetNdivisions(505)
    h.GetYaxis().SetNdivisions(510)


def _draw_headers(extra_left=None):
    lx = ROOT.TLatex()
    lx.SetTextSize(0.035)
    lx.SetTextFont(42)
    lx.SetTextAlign(13)
    lx.DrawLatexNDC(0.16, 0.95, extra_left if extra_left is not None else HEADER_LEFT)


def plot_hists_1d(hists, labels, outname, x_title="", y_title="entries",
                   colors=None, logy=False, line_width=3, y_factor=1.25,
                   canvas_size=(800, 700), extra_left=None,
                   fit_func=None, annotation=None):
    """Overlay one or more 1D histograms on a single canvas; saves PNG+PDF.
    fit_func: an already-fitted ROOT.TF1, drawn on top as a dashed curve.
    annotation: extra text line(s) drawn below the header (e.g. fit results)."""
    if not hists:
        return
    ROOT.gStyle.SetOptStat(0)
    colors = colors or [ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2,
                         ROOT.kMagenta + 1, ROOT.kOrange + 7, ROOT.kBlack]

    hists_draw = []
    for i, h in enumerate(hists):
        hc = h.Clone(f"{h.GetName()}_plot_{i}")
        hc.SetDirectory(0)
        hc.SetStats(0)
        hc.SetLineColor(colors[i % len(colors)])
        hc.SetLineWidth(line_width)
        hists_draw.append(hc)

    c = ROOT.TCanvas(f"c_{outname.replace('/', '_')}", "", *canvas_size)
    c.SetTopMargin(0.10)
    c.SetBottomMargin(0.12)
    c.SetLeftMargin(0.14)
    c.SetRightMargin(0.05)
    if logy:
        c.SetLogy()

    max_y = max(hc.GetMaximum() for hc in hists_draw)
    hists_draw[0].SetTitle("")
    hists_draw[0].GetXaxis().SetTitle(x_title)
    hists_draw[0].GetYaxis().SetTitle(y_title)
    hists_draw[0].SetMaximum(max_y * y_factor if max_y > 0 else 1.0)
    hists_draw[0].SetMinimum(0.0)
    _style_axes(hists_draw[0])
    hists_draw[0].Draw("HIST")
    for hc in hists_draw[1:]:
        hc.Draw("HIST SAME")

    if fit_func is not None:
        fit_func.SetLineColor(ROOT.kBlack)
        fit_func.SetLineStyle(2)
        fit_func.SetLineWidth(2)
        fit_func.Draw("SAME")

    if labels:
        leg = ROOT.TLegend(0.65, 0.90 - 0.05 * len(labels), 0.94, 0.90)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.03)
        for hc, lab in zip(hists_draw, labels):
            leg.AddEntry(hc, lab, "l")
        leg.Draw()

    _draw_headers(extra_left)

    if annotation:
        lx = ROOT.TLatex()
        lx.SetTextSize(0.032)
        lx.SetTextFont(42)
        lx.SetTextAlign(13)
        for i, line in enumerate(annotation.split("\n")):
            lx.DrawLatexNDC(0.17, 0.87 - i * 0.045, line)

    c.Modified()
    c.Update()
    c.SaveAs(f"{outname}.png")
    c.SaveAs(f"{outname}.pdf")


def plot_hist_1d(hist, outname, x_title="", y_title="entries", **kwargs):
    """Single-histogram convenience wrapper around plot_hists_1d."""
    plot_hists_1d([hist], [], outname, x_title=x_title, y_title=y_title, **kwargs)


def plot_hist_2d(hist, outname, x_title="", y_title="", z_title="",
                  canvas_size=(900, 750), logz=False, extra_left=None):
    """2D COLZ plot with the same header convention as the 1D plots."""
    ROOT.gStyle.SetOptStat(0)
    h = hist.Clone(f"{hist.GetName()}_plot2d")
    h.SetDirectory(0)
    h.SetTitle("")
    h.GetXaxis().SetTitle(x_title)
    h.GetYaxis().SetTitle(y_title)
    h.GetZaxis().SetTitle(z_title)
    _style_axes(h)
    h.GetZaxis().SetTitleSize(0.04)
    h.GetZaxis().SetLabelSize(0.035)

    c = ROOT.TCanvas(f"c_{outname.replace('/', '_')}", "", *canvas_size)
    c.SetTopMargin(0.10)
    c.SetBottomMargin(0.12)
    c.SetLeftMargin(0.13)
    c.SetRightMargin(0.17)
    if logz:
        c.SetLogz()
    h.Draw("COLZ")

    _draw_headers(extra_left)
    c.Modified()
    c.Update()
    c.SaveAs(f"{outname}.png")
    c.SaveAs(f"{outname}.pdf")
