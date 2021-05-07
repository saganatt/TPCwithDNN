"""Plotting functions for optimization results"""
# pylint: disable=missing-function-docstring, missing-class-docstring
# pylint: disable=fixme
import datetime

import numpy as np
from root_numpy import fill_hist # pylint: disable=import-error

from ROOT import gROOT, gStyle, gPad # pylint: disable=import-error, no-name-in-module
from ROOT import TCavas, TLegend, TLatex, TPaveText # pylint: disable=import-error, no-name-in-module
from ROOT import TH1F, TH2F, TFile # pylint: disable=import-error, no-name-in-module
from ROOT import kWhite, kBlue, kGreen, kRed, kCyan, kOrange, kMagenta # pylint: disable=import-error, no-name-in-module

"Global ROOT configuration, loaded only once"
gROOT.SetStyle("Plain")
gROOT.SetBatch()
gStyle.SetOptStat(0)
gStyle.SetOptTitle(0)
gStyle.SetTextFont(42)
gStyle.SetLabelFont(42, "xyz")
gStyle.SetTitleFont(42, "xyz")
gStyle.SetPalette(kDarkBodyRadiator)

# Utilities for filling histograms for apply
def create_apply_histos(config, suffix, infix=""):
    h_dist = TH2F("%s_%s%s" % (config.h_dist_name, infix, suffix),
                  "", 500, -5, 5, 500, -5, 5)
    h_deltas = TH1F("%s_%s%s" % (config.h_deltas_name, infix, suffix),  "", 1000, -1., 1.)
    h_deltas_vs_dist = TH2F("%s_%s%s" % \
                            (config.h_deltas_vs_dist_name, infix, suffix),
                            "", 500, -5.0, 5.0, 100, -0.5, 0.5)
    return h_dist, h_deltas, h_deltas_vs_dist

def fill_std_dev_apply_hist(h_deltas_vs_dist, hist_name, suffix, infix=""):
    h1tmp = h_deltas_vs_dist.ProjectionX("h1tmp")
    h_std_dev = h1tmp.Clone("%s_%s%s" % (hist_name, infix, suffix))
    h_std_dev.Reset()
    h_std_dev.SetXTitle("d#it{%s}_{true} (cm)")
    h_std_dev.SetYTitle("std.dev. of (d#it{%s}_{pred} - d#it{%s}_{true}) (cm)")
    nbin = int(h_std_dev.GetNbinsX())
    for ibin in range(0, nbin):
        h1diff = h_deltas_vs_dist.ProjectionY("h1diff", ibin+1, ibin+1, "")
        stddev = h1diff.GetStdDev()
        stddev_err = h1diff.GetStdDevError()
        h_std_dev.SetBinContent(ibin+1, stddev)
        h_std_dev.SetBinError(ibin+1, stddev_err)
    h_std_dev.Write()

def fill_profile_apply_hist(h_deltas_vs_dist_all_events, profile_name, suffix):
    prof_all_events = h_deltas_vs_dist_all_events.ProfileX()
    prof_all_events.SetName("%s_all_events_%s" % (profile_name, suffix))
    prof_all_events.Write()

def fill_apply_tree_single_event(config, indexev, distortion_numeric_flat_m,
                                 distortion_predict_flat_m, deltas_flat_a, deltas_flat_m):
    h_suffix = "Ev%d_Mean%d_%s" % (indexev[0], indexev[1], config.suffix)
    h_dist, h_deltas, h_deltas_vs_dist = create_apply_histos(config, h_suffix)

    fill_hist(h_dist, np.concatenate((distortion_numeric_flat_m,
                                      distortion_predict_flat_m), axis=1))
    fill_hist(h_deltas, deltas_flat_a)
    fill_hist(h_deltas_vs_dist,
              np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))

    prof = h_deltas_vs_dist.ProfileX()
    prof.SetName("%s_%s" % (config.profile_name, h_suffix))

    h_dist.Write()
    h_deltas.Write()
    h_deltas_vs_dist.Write()
    prof.Write()
    fill_std_dev_apply_hist(h_deltas_vs_dist, config.h_std_dev_name, h_suffix)

def fill_apply_tree(h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events,
                    distortion_numeric_flat_m, distortion_predict_flat_m,
                    deltas_flat_a, deltas_flat_m):
    fill_hist(h_dist_all_events, np.concatenate((distortion_numeric_flat_m, \
                                                 distortion_predict_flat_m), axis=1))
    fill_hist(h_deltas_all_events, deltas_flat_a)
    fill_hist(h_deltas_vs_dist_all_events,
              np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))


def get_apply_results_single_event(pred_outputs, exp_outputs):
    distortion_predict_group = pred_outputs
    distortion_predict_flat_m = distortion_predict_group.reshape(-1, 1)
    distortion_predict_flat_a = distortion_predict_group.flatten()

    distortion_numeric_group = exp_outputs
    distortion_numeric_flat_m = distortion_numeric_group.reshape(-1, 1)
    distortion_numeric_flat_a = distortion_numeric_group.flatten()

    deltas_flat_a = (distortion_predict_flat_a - distortion_numeric_flat_a)
    deltas_flat_m = (distortion_predict_flat_m - distortion_numeric_flat_m)

    return distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m


# Utilities to plot apply results
def plot(config):
    gROOT.ForceStyle()
    sel_opts = np.array(config.opt_predout)
    sel_opts_names = np.array(config.nameopt_predout)
    sel_opts_names = sel_opts_names[sel_opts == 1]
    for opt_name in sel_opts_names:
        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (config.dirval, config.suffix,
                             config.train_events), "open")
        h_dist_all_events = myfile.Get("%s_all_events_%s" % (config.h_dist_name,
                                                             config.suffix))
        h_deltas_all_events = myfile.Get("%s_all_events_%s" % \
                                         (config.h_deltas_name, config.suffix))
        h_deltas_vs_dist_all_events = myfile.Get("%s_all_events_%s" % \
                                                 (config.h_deltas_vs_dist_name,
                                                  config.suffix))
        profile_deltas_vs_dist_all_events = \
            myfile.Get("%s_all_events_%s" % (config.profile_name, config.suffix))

        plot_distortion(config, h_dist_all_events, h_deltas_all_events,
                        h_deltas_vs_dist_all_events,
                        profile_deltas_vs_dist_all_events,
                        config.suffix, opt_name)

        if config.per_event_hists:
            counter = 0
            for iexperiment in config.partition['apply']:
                h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], config.suffix)
                h_dist = myfile.Get("%s_%s" % (config.h_dist_name, h_suffix))
                h_deltas = myfile.Get("%s_%s" % (config.h_deltas_name, h_suffix))
                h_deltas_vs_dist = myfile.Get("%s_%s" % (config.h_deltas_vs_dist_name, h_suffix))
                profile = myfile.Get("%s_%s" % (config.profile_name, h_suffix))
                plot_distortion(config, h_dist, h_deltas, h_deltas_vs_dist, profile,
                                h_suffix, opt_name)
                counter = counter + 1
                if counter > 100:
                    return

def plot_distortion(config, h_dist, h_deltas, h_deltas_vs_dist, prof, suffix, opt_name):
    cev = TCanvas("canvas_%s_nEv%d_%s" % (suffix, config.train_events, opt_name),
                  "canvas_%s_nEv%d_%s" % (suffix, config.train_events, opt_name),
                  800, 800) # 1600 x 1600
    cev.Divide(2, 2)
    c1 = cev.cd(1)
    c1.SetMargin(0.12, 0.12, 0.12, 0.05)
    gPad.SetLogz()
    setup_frame(h_dist, "d#it{%s}_{true} (cm)" % opt_name.lower(),
                "d#it{%s}_{pred} (cm)" % opt_name.lower(),
                x_offset=1.2, y_offset=1.2, label_size=0.04)
    h_dist.Draw("colz")
    txt1 = add_desc(config, 0.18, 0.7, 0.3, 0.9, 0.04,
                    {"add_alice": False, "add_gran": True, "add_inputs": False,
                     "add_events": True})
    txt1.Draw()
    c2 = cev.cd(2)
    c2.SetMargin(0.12, 0.05, 0.12, 0.05)
    gPad.SetLogy()
    setup_frame(h_deltas_vs_dist, "d#it{%s}_{true} (cm)" % opt_name.lower(),
                "Entries", x_offset=1.2, y_offset=1.2, label_size=0.04)
    h_deltas_vs_dist.ProjectionX().Draw()
    txt2 = add_desc(config, 0.18, 0.7, 0.3, 0.9, 0.04,
                    {"add_alice": False, "add_gran": True, "add_inputs": False,
                     "add_events": True})
    txt2.Draw()
    c3 = cev.cd(3)
    c3.SetMargin(0.12, 0.05, 0.12, 0.05)
    gPad.SetLogy()
    setup_frame(h_deltas, "<d#it{%s}_{pred} - d#it{%s}_{true}> (cm)" %\
                                (opt_name.lower(), opt_name.lower()),
                "Entries", x_offset=1.2, y_offset=1.5, label_size=0.04)
    h_deltas.Draw()
    txt3 = add_desc(config, 0.18, 0.7, 0.3, 0.9, 0.04,
                    {"add_alice": False, "add_gran": True, "add_inputs": False,
                     "add_events": True})
    txt3.Draw()
    c4 = cev.cd(4)
    c4.SetMargin(0.15, 0.05, 0.12, 0.05)
    setup_frame(prof, "d#it{%s}_{true} (cm)" % opt_name.lower(),
                "<d#it{%s}_{pred} - d#it{%s}_{true}> (cm)" %\
                        (opt_name.lower(), opt_name.lower()),
                x_offset=1.2, y_offset=1.8, label_size=0.04)
    prof.Draw()
    txt4 = add_desc(config, 0.45, 0.7, 0.85, 0.9, 0.04,
                    {"add_alice": False, "add_gran": True, "add_inputs": False,
                     "add_events": True})
    txt4.Draw()
    cev.SaveAs("%s/canvas_%s_nEv%d.pdf" % (config.dirplots, suffix,
                                           config.train_events))

def setup_canvas(suffix, hist_name, opt_name):
    full_name = "%s_canvas_%s_%s" % (hist_name, suffix, opt_name)
    canvas = TCanvas(full_name, full_name, 0, 0, 800, 800)
    canvas.SetMargin(0.12, 0.05, 0.12, 0.05) # left, right, bottom, top
    canvas.SetTicks(1, 1)
    return canvas

def save_canvas(config, canvas, frame, prefix, func_name, file_formats):
    file_name = "%s_wide_%s_%s" % (prefix, func_name, config.suffix)
    for file_format in file_formats:
        canvas.SaveAs("%s.%s" % (file_name, file_format))
    frame.GetYaxis().SetRangeUser(-0.05, +0.05)
    file_name = "%s_zoom_%s_%s" % (prefix, func_name, config.suffix)
    for file_format in file_formats:
        canvas.SaveAs("%s.%s" % (file_name, file_format))

def setup_text(xmin, ymin, xmax, ymax, text_size):
    txt = TPaveText(xmin, ymin, xmax, ymax, "NDC")
    txt.SetFillColor(kWhite)
    txt.SetFillStyle(0)
    txt.SetBorderSize(0)
    txt.SetTextAlign(12) # middle,left
    txt.SetTextFont(42) # helvetica
    txt.SetTextSize(size)
    return txt

def add_desc(config, content):
    if content["add_alice"]:
        txt.AddText("ALICE work in progress")
    if content["add_gran"]:
        gran_desc = "#it{n}_{#it{#varphi}} #times #it{n}_{#it{r}} #times #it{n}_{#it{z}}"
        gran_str = "%d #times %d #times %d" % (config.grid_phi, config.grid_r,
                                               config.grid_z)
        txt.AddText("%s = %s" % (gran_desc, gran_str))
    if content["add_inputs"]:
        if config.opt_train[0] == 1 and config.opt_train[1] == 1:
            txt.AddText("inputs: #it{#rho}_{SC} - <#it{#rho}_{SC}>, <#it{#rho}_{SC}>")
        elif config.opt_train[1] == 1:
            txt.AddText("inputs: #it{#rho}_{SC} - <#it{#rho}_{SC}>")
    if content["add_events"]:
        txt.AddText("#it{N}_{ev}^{training} = %d" % config.train_events)
    if config.name == "dnn":
        txt.AddText("%d epochs" % config.epochs)
    return txt

def draw_multievent_hist(config, events_counts, func_label, hist_name, source_hist):
    gROOT.ForceStyle()
    gran_str = "%d#times %d #times %d" % (config.grid_phi, config.grid_r,
                                          config.grid_z)
    date = datetime.date.today().strftime("%Y%m%d")

    canvas = TCanvas()
    canvas.SetMargin(0.12, 0.05, 0.12, 0.05) # left, right, bottom, top
    canvas.SetTicks(1, 1)
    # canvas = setup_canvas(suffix, hist_name, opt_name)
    frame = canvas.DrawFrame(-5, -0.5, +5, +0.5)
    setup_frame(frame, "", "", x_offset=1.5, y_offset=1.5, label_size=0.04)
    leg = setup_legend(x1=0.5, y1=0.7, x2=0.9, y2=0.9, text_size=0.03)

    file_formats = ["pdf", "png"]
    # file_formats = ["png", "eps", "pdf"]
    var_labels = np.array(["r", "r#varphi", "z"])
    colors = [kBlue+1, kGreen+2, kRed+1, kCyan+2, kOrange+7, kMagenta+2]
    #colors = [kRed+1, kMagenta+2, kOrange+7, kCyan+1, kMagenta+2]
    sel_opts = np.array(config.opt_predout)
    sel_opts_names = np.array(config.nameopt_predout)
    sel_opts_names = sel_opts_names[sel_opts == 1]
    sel_var_labels = var_labels[sel_opts == 1]
    for opt_name, var_label in zip(sel_opts_names, sel_var_labels):
        x_label = "d#it{%s}_{true} (cm)" % var_label
        y_label = "%s of d#it{%s}_{pred} - d#it{%s}_{true} (cm)" %\
                  (func_label, var_label, var_label)
        frame.GetXaxis().SetTitle(x_label)
        frame.GetYaxis().SetTitle(y_label)

        # TODO: Clean these codes
        for i, (train_events, _, _, _) in enumerate(events_counts):
            filename = "%s/output_%s_nEv%d.root" % (config.dirval, config.suffix, train_events)
            config.logger.info("Reading %s...", filename)

            root_file = TFile.Open(filename, "read")
            hist = root_file.Get("%s_all_events_%s" % (source_hist, config.suffix))
            hist.SetDirectory(0)
            hist.Draw("same")
            hist.SetMarkerStyle(20)
            hist.SetMarkerColor(colors[i])
            hist.SetLineColor(colors[i])
            # train_events_k = train_events / 1000
            leg.AddEntry(hist, "%d, %s" % (train_events, gran_str), "LP")

            if "mean" in hist_name and "std" in hist_name:
                hist.Delete("C")
                leg.DeleteEntry()
                hist_mean = root_file.Get("%s_all_events_%s" % \
                        (config.profile_name, config.suffix))
                hist_stddev = root_file.Get("%s_all_events_%s" % \
                        (config.h_std_dev_name, config.suffix))
                hist_mean.SetDirectory(0)
                hist_stddev.SetDirectory(0)
                hist = hist_mean.ProjectionX("hist_meanSD")
                hist.Reset()
                hist.Sumw2()
                hist.SetDirectory(0)
                nbin = hist_mean.GetNbinsX()
                for ibin in range(0,nbin):
                    hist.SetBinContent(ibin+1, hist_mean.GetBinContent(ibin+1))
                    hist.SetBinError(ibin+1, hist_stddev.GetBinContent(ibin+1))

                hist.SetMarkerStyle(20)
                hist.SetMarkerColor(colors[i])
                hist.SetLineColor(colors[i])
                hist.SetFillColor(colors[i])
                hist.SetFillStyle(3001)
                hist.Draw("sameE2")
                leg.AddEntry(hist, "%d, %s" % (train_events, gran_str), "FP")

            root_file.Close()

        leg.Draw()
        txt = add_desc(config, 0.15, 0.81, 0.4, 0.89, 0.03,
                       {"add_alice": False, "add_gran": False, "add_inputs": True,
                        "add_events": False})
        txt.Draw()
        save_canvas(config, canvas, frame, "%s/%s" % (config.dirplots, date),
                    hist_name, file_formats)

def draw_mean(config, events_counts):
    draw_multievent_hist(config, events_counts, "#it{#mu}", "mean", config.profile_name)

def draw_std_dev(config, events_counts):
    draw_multievent_hist(config, events_counts, "#it{#sigma}_{std}", "std_dev",
                         config.h_std_dev_name)

def draw_mean_std_dev(config, events_counts):
    draw_multievent_hist(config, events_counts, "#it{#mu} #pm #it{#sigma}_{std}",
                         "mean_std_dev", config.profile_name)

def setup_frame(frame, x_label, y_label, z_label=None, x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.04, title_size=None):
    if frame is None:
        frame = gPad.GetPrimitive("htemp")
    if title_size is None:
        title_size = label_size

    frame.GetXaxis().SetTitle(x_label)
    frame.GetXaxis().SetTitleOffset(x_offset)
    frame.GetXaxis().CenterTitle(True)
    frame.GetXaxis().SetTitleSize(label_size)
    frame.GetXaxis().SetLabelSize(label_size)

    frame.GetYaxis().SetTitle(y_label)
    frame.GetYaxis().SetTitleOffset(y_offset)
    frame.GetYaxis().CenterTitle(True)
    frame.GetYaxis().SetTitleSize(label_size)
    frame.GetYaxis().SetLabelSize(label_size)

    if z_label is not None:
        frame.GetZaxis().SetTitle(z_label)
        frame.GetZaxis().SetTitleOffset(z_offset)
        frame.GetZaxis().CenterTitle(True)
        frame.GetZaxis().SetTitleSize(label_size)
        frame.GetZaxis().SetLabelSize(label_size)

def setup_legend(x1, y1, x2, y2, text_size=0.03, ncols=None, margin=None):
    leg = TLegend(x1, y1, x2, y2)
    if ncols is not None:
        leg.SetNColumns(ncols)
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextSize(text_size)
    if margin is not None:
        leg.SetMargin(margin)
    leg.SetHeader("Train setup: #it{N}_{ev}^{training}, #it{n}_{#it{#varphi}}" +\
                  " #times #it{n}_{#it{r}} #times #it{n}_{#it{z}}", "C")

    return leg

def add_alice_text(x, y, text_size=0.04):
    tex = TLatex(x, y, "#scale[0.8]{ALICE work in progress}")
    tex.SetTextSize(text_size)
    tex.SetNDC()
    return tex
