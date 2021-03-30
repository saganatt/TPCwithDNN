# pylint: disable=too-many-instance-attributes, too-many-statements, too-many-arguments, fixme
# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=protected-access, too-many-locals, wrong-import-position
import os
import datetime

SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_DETERMINISTIC_OPS'] = "1"
# Make TF using 1 thread on CPU
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

import random
random.seed(SEED)

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import plot_model

from root_numpy import fill_hist # pylint: disable=import-error
from ROOT import TH1F, TH2F, TFile, TCanvas, TLegend, TPaveText, gPad # pylint: disable=import-error, no-name-in-module
from ROOT import gStyle, kWhite, kBlue, kGreen, kRed, kCyan, kOrange, kMagenta # pylint: disable=import-error, no-name-in-module
from ROOT import gROOT  # pylint: disable=import-error, no-name-in-module

from tpcwithdnn.symmetry_padding_3d import SymmetryPadding3d
from tpcwithdnn.logger import get_logger
from tpcwithdnn.fluctuation_data_generator import FluctuationDataGenerator
from tpcwithdnn.utilities_dnn import u_net
from tpcwithdnn.data_loader import load_train_apply, get_event_mean_indices

matplotlib.use("Agg")

class DnnOptimiser:
    # Class Attribute
    species = "dnnoptimiser"

    h_dist_name = "h_dist"
    h_deltas_name = "h_deltas"
    h_deltas_vs_dist_name = "h_deltas_vs_dist"
    profile_name = "profile_deltas_vs_dist"
    h_std_dev_name = "h_std_dev"

    def __init__(self, data_param, case):
        self.logger = get_logger()
        self.logger.info("DnnOptimizer::Init\nCase: %s", case)

        # Dataset config
        self.grid_phi = data_param["grid_phi"]
        self.grid_z = data_param["grid_z"]
        self.grid_r = data_param["grid_r"]

        self.selopt_input = data_param["selopt_input"]
        self.selopt_output = data_param["selopt_output"]
        self.opt_train = data_param["opt_train"]
        self.opt_predout = data_param["opt_predout"]
        self.nameopt_predout = data_param["nameopt_predout"]
        self.dim_input = sum(self.opt_train)
        self.dim_output = sum(self.opt_predout)
        self.use_scaler = data_param["use_scaler"]

        # Directories
        self.dirmodel = data_param["dirmodel"]
        self.dirval = data_param["dirval"]
        self.dirplots = data_param["dirplots"]
        train_dir = data_param["dirinput_bias"] if data_param["train_bias"] \
                    else data_param["dirinput_nobias"]
        test_dir = data_param["dirinput_bias"] if data_param["test_bias"] \
                    else data_param["dirinput_nobias"]
        apply_dir = data_param["dirinput_bias"] if data_param["apply_bias"] \
                    else data_param["dirinput_nobias"]
        self.dirinput_train = "%s/SC-%d-%d-%d/" % \
                              (train_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_test = "%s/SC-%d-%d-%d/" % \
                             (test_dir, self.grid_z, self.grid_r, self.grid_phi)
        self.dirinput_apply = "%s/SC-%d-%d-%d/" % \
                              (apply_dir, self.grid_z, self.grid_r, self.grid_phi)

        # DNN config
        self.filters = data_param["filters"]
        self.pooling = data_param["pooling"]
        self.batch_size = data_param["batch_size"]
        self.shuffle = data_param["shuffle"]
        self.depth = data_param["depth"]
        self.batch_normalization = data_param["batch_normalization"]
        self.dropout = data_param["dropout"]
        self.epochs = data_param["epochs"]
        self.lossfun = data_param["lossfun"]
        if data_param["metrics"] == "rmse":
            self.metrics = "root_mean_squared_error"
        else:
            self.metrics = data_param["metrics"]
        self.adamlr = data_param["adamlr"]

        self.params = {'phi_slice': self.grid_phi,
                       'r_row' : self.grid_r,
                       'z_col' : self.grid_z,
                       'batch_size': self.batch_size,
                       'shuffle': self.shuffle,
                       'opt_train' : self.opt_train,
                       'opt_predout' : self.opt_predout,
                       'selopt_input' : self.selopt_input,
                       'selopt_output' : self.selopt_output,
                       'use_scaler': self.use_scaler}

        self.suffix = "phi%d_r%d_z%d_filter%d_poo%d_drop%.2f_depth%d_batch%d_scaler%d" % \
                (self.grid_phi, self.grid_r, self.grid_z, self.filters, self.pooling,
                 self.dropout, self.depth, self.batch_normalization, self.use_scaler)
        self.suffix = "%s_useSCMean%d_useSCFluc%d" % \
                (self.suffix, self.opt_train[0], self.opt_train[1])
        self.suffix = "%s_pred_doR%d_dophi%d_doz%d" % \
                (self.suffix, self.opt_predout[0], self.opt_predout[1], self.opt_predout[2])
        self.suffix_ds = "phi%d_r%d_z%d" % \
                (self.grid_phi, self.grid_r, self.grid_z)

        if not os.path.isdir(self.dirmodel):
            os.makedirs(self.dirmodel)

        if not os.path.isdir(self.dirval):
            os.makedirs(self.dirval)

        if not os.path.isdir(self.dirplots):
            os.makedirs(self.dirplots)

        self.logger.info("I am processing the configuration %s", self.suffix)
        if self.dim_output > 1:
            self.logger.fatal("YOU CAN PREDICT ONLY 1 DISTORSION. The sum of opt_predout == 1")
        self.logger.info("Inputs active for training: (SCMean, SCFluctuations)=(%d, %d)",
                         self.opt_train[0], self.opt_train[1])

        # Parameters for getting input indices
        self.maxrandomfiles = data_param["maxrandomfiles"]
        self.range_mean_index = data_param["range_mean_index"]
        self.indices_events_means = None
        self.partition = None
        self.total_events = 0
        self.train_events = 0
        self.test_events = 0
        self.apply_events = 0

        gROOT.SetStyle("Plain")
        gROOT.SetBatch()
        gStyle.SetOptStat(0)
        gStyle.SetTextFont(42)
        gStyle.SetLabelFont(42, "xyz")
        gStyle.SetTitleFont(42, "xyz")

    def train(self):
        self.logger.info("DnnOptimizer::train")

        training_generator = FluctuationDataGenerator(self.partition['train'],
                                                      data_dir=self.dirinput_train, **self.params)
        validation_generator = FluctuationDataGenerator(self.partition['validation'],
                                                        data_dir=self.dirinput_test, **self.params)
        model = u_net((self.grid_phi, self.grid_r, self.grid_z, self.dim_input),
                      depth=self.depth, batchnorm=self.batch_normalization,
                      pool_type=self.pooling, start_channels=self.filters, dropout=self.dropout)
        if self.metrics == "root_mean_squared_error":
            metrics = RootMeanSquaredError()
        else:
            metrics = self.metrics
        model.compile(loss=self.lossfun, optimizer=Adam(lr=self.adamlr),
                      metrics=[metrics]) # Mean squared error

        model.summary()
        plot_model(model, to_file='%s/model_%s_nEv%d.png' % \
                   (self.dirplots, self.suffix, self.train_events),
                   show_shapes=True, show_layer_names=True)

        #log_dir = "logs/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_dir = 'logs/' + '%s_nEv%d' % (self.suffix, self.train_events)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        model._get_distribution_strategy = lambda: None
        his = model.fit(training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        epochs=self.epochs, callbacks=[tensorboard_callback],
                        shuffle=False)

        plt.style.use("ggplot")
        plt.figure()
        plt.yscale('log')
        plt.plot(np.arange(0, self.epochs), his.history["loss"], label="train_loss")
        plt.plot(np.arange(0, self.epochs), his.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), his.history[self.metrics],
                 label="train_" + self.metrics)
        plt.plot(np.arange(0, self.epochs), his.history["val_" + self.metrics],
                 label="val_" + self.metrics)
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("%s/plot_%s_nEv%d.png" % (self.dirplots, self.suffix, self.train_events))

        model_json = model.to_json()
        with open("%s/model_%s_nEv%d.json" % (self.dirmodel, self.suffix, self.train_events), "w") \
            as json_file:
            json_file.write(model_json)
        model.save_weights("%s/model_%s_nEv%d.h5" % (self.dirmodel, self.suffix, self.train_events))
        self.logger.info("Saved trained model to disk")


    def apply(self):
        self.logger.info("DnnOptimizer::apply, input size: %d", self.dim_input)

        json_file = open("%s/model_%s_nEv%d.json" % \
                         (self.dirmodel, self.suffix, self.train_events), "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = \
            model_from_json(loaded_model_json, {'SymmetryPadding3d' : SymmetryPadding3d})
        loaded_model.load_weights("%s/model_%s_nEv%d.h5" % \
                                  (self.dirmodel, self.suffix, self.train_events))

        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (self.dirval, self.suffix, self.train_events), "recreate")
        h_dist_all_events = TH2F("%s_all_events_%s" % (self.h_dist_name, self.suffix),
                                 "", 500, -5, 5, 500, -5, 5)
        h_deltas_all_events = TH1F("%s_all_events_%s" % (self.h_deltas_name, self.suffix),
                                   "", 1000, -1., 1.)
        h_deltas_vs_dist_all_events = TH2F("%s_all_events_%s" % \
                                           (self.h_deltas_vs_dist_name, self.suffix),
                                           "", 500, -5.0, 5.0, 100, -0.5, 0.5)

        for iexperiment in self.partition['apply']:
            indexev = iexperiment
            inputs_, exp_outputs_ = load_train_apply(self.dirinput_apply, indexev,
                                                     self.selopt_input, self.selopt_output,
                                                     self.grid_r, self.grid_phi, self.grid_z,
                                                     self.opt_train, self.opt_predout)
            inputs_single = np.empty((1, self.grid_phi, self.grid_r, self.grid_z, self.dim_input))
            exp_outputs_single = np.empty((1, self.grid_phi, self.grid_r,
                                           self.grid_z, self.dim_output))
            inputs_single[0, :, :, :, :] = inputs_
            exp_outputs_single[0, :, :, :, :] = exp_outputs_

            distortion_predict_group = loaded_model.predict(inputs_single)
            distortion_predict_flat_m = distortion_predict_group.reshape(-1, 1)
            distortion_predict_flat_a = distortion_predict_group.flatten()

            distortion_numeric_group = exp_outputs_single
            distortion_numeric_flat_m = distortion_numeric_group.reshape(-1, 1)
            distortion_numeric_flat_a = distortion_numeric_group.flatten()
            deltas_flat_a = (distortion_predict_flat_a - distortion_numeric_flat_a)
            deltas_flat_m = (distortion_predict_flat_m - distortion_numeric_flat_m)

            h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], self.suffix)
            h_dist = TH2F("%s_%s" % (self.h_dist_name, h_suffix), "", 500, -5, 5, 500, -5, 5)
            h_deltas = TH1F("%s_%s" % (self.h_deltas_name, h_suffix), "", 1000, -1., 1.)
            h_deltas_vs_dist = TH2F("%s_%s" % (self.h_deltas_vs_dist_name, h_suffix), "",
                                    500, -5.0, 5.0, 100, -0.5, 0.5)

            fill_hist(h_dist_all_events, np.concatenate((distortion_numeric_flat_m, \
                                distortion_predict_flat_m), axis=1))
            fill_hist(h_dist, np.concatenate((distortion_numeric_flat_m,
                                              distortion_predict_flat_m), axis=1))
            fill_hist(h_deltas, deltas_flat_a)
            fill_hist(h_deltas_all_events, deltas_flat_a)
            fill_hist(h_deltas_vs_dist,
                      np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))
            fill_hist(h_deltas_vs_dist_all_events,
                      np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))

            prof = h_deltas_vs_dist.ProfileX()
            prof.SetName("%s_%s" % (self.profile_name, h_suffix))

            h_dist.Write()
            h_deltas.Write()
            h_deltas_vs_dist.Write()
            prof.Write()

            h1tmp = h_deltas_vs_dist.ProjectionX("h1tmp")
            h_std_dev = h1tmp.Clone("%s_%s" % (self.h_std_dev_name, h_suffix))
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

        h_dist_all_events.Write()
        h_deltas_all_events.Write()
        h_deltas_vs_dist_all_events.Write()
        prof_all_events = h_deltas_vs_dist_all_events.ProfileX()
        prof_all_events.SetName("%s_all_events_%s" % (self.profile_name, self.suffix))
        prof_all_events.Write()

        h1tmp = h_deltas_vs_dist_all_events.ProjectionX("h1tmp")
        h_std_dev_all_events = h1tmp.Clone("%s_all_events_%s" % (self.h_std_dev_name, self.suffix))
        h_std_dev_all_events.Reset()
        h_std_dev_all_events.SetXTitle("d#it{%s}_{true} (cm)")
        h_std_dev_all_events.SetYTitle("std.dev. of (d#it{%s}_{pred} - d#it{%s}_{true}) (cm)")
        nbin = int(h_std_dev_all_events.GetNbinsX())
        for ibin in range(0, nbin):
            h1diff = h_deltas_vs_dist_all_events.ProjectionY("h1diff", ibin+1, ibin+1, "")
            stddev = h1diff.GetStdDev()
            stddev_err = h1diff.GetStdDevError()
            h_std_dev_all_events.SetBinContent(ibin+1, stddev)
            h_std_dev_all_events.SetBinError(ibin+1, stddev_err)
        h_std_dev_all_events.Write()

        myfile.Close()
        self.logger.info("Done apply")


    def plot_distorsion(self, h_dist, h_deltas, h_deltas_vs_dist, prof, suffix, opt_name):
        cev = TCanvas("canvas_%s_nEv%d_%s" % (suffix, self.train_events, opt_name),
                      "canvas_%s_nEv%d_%s" % (suffix, self.train_events, opt_name),
                      1600, 1600)
        cev.Divide(2, 2)
        c1 = cev.cd(1)
        c1.SetMargin(0.12, 0.12, 0.12, 0.05)
        gPad.SetLogz()
        h_dist.GetXaxis().SetTitle("d#it{%s}_{true} (cm)" % opt_name.lower())
        h_dist.GetYaxis().SetTitle("d#it{%s}_{pred} (cm)" % opt_name.lower())
        h_dist.GetXaxis().CenterTitle(True)
        h_dist.GetYaxis().CenterTitle(True)
        h_dist.GetXaxis().SetTitleOffset(1.2)
        h_dist.GetYaxis().SetTitleOffset(1.2)
        h_dist.Draw("colz")
        txt1 = self.add_desc_to_canvas(0.18, 0.7, 0.3, 0.9, 0.04, True, False, True, False)
        txt1.Draw()
        c2 = cev.cd(2)
        c2.SetMargin(0.12, 0.05, 0.12, 0.05)
        gPad.SetLogy()
        h_deltas_vs_dist.GetXaxis().SetTitle("d#it{%s}_{true} (cm)" % opt_name.lower())
        h_deltas_vs_dist.GetYaxis().SetTitle("Entries")
        h_deltas_vs_dist.GetXaxis().CenterTitle(True)
        h_deltas_vs_dist.GetYaxis().CenterTitle(True)
        h_deltas_vs_dist.GetXaxis().SetTitleOffset(1.2)
        h_deltas_vs_dist.GetYaxis().SetTitleOffset(1.2)
        h_deltas_vs_dist.ProjectionX().Draw()
        txt2 = self.add_desc_to_canvas(0.18, 0.7, 0.3, 0.9, 0.04, True, False, True, False)
        txt2.Draw()
        c3 = cev.cd(3)
        c3.SetMargin(0.12, 0.05, 0.12, 0.05)
        gPad.SetLogy()
        h_deltas.GetXaxis().SetTitle("<d#it{%s}_{pred} - d#it{%s}_{true}> (cm)"
                                     % (opt_name.lower(), opt_name.lower()))
        h_deltas.GetYaxis().SetTitle("Entries")
        h_deltas.GetXaxis().CenterTitle(True)
        h_deltas.GetYaxis().CenterTitle(True)
        h_deltas.GetXaxis().SetTitleOffset(1.2)
        h_deltas.GetYaxis().SetTitleOffset(1.5)
        h_deltas.Draw()
        txt3 = self.add_desc_to_canvas(0.18, 0.7, 0.3, 0.9, 0.04, True, False, True, False)
        txt3.Draw()
        c4 = cev.cd(4)
        c4.SetMargin(0.15, 0.05, 0.12, 0.05)
        prof.GetYaxis().SetTitle("<d#it{%s}_{pred} - d#it{%s}_{true}> (cm)"
                                 % (opt_name.lower(), opt_name.lower()))
        prof.GetYaxis().SetTitleOffset(1.3)
        prof.GetXaxis().SetTitle("d#it{%s}_{true} (cm)" % opt_name.lower())
        prof.GetXaxis().CenterTitle(True)
        prof.GetYaxis().CenterTitle(True)
        prof.GetXaxis().SetTitleOffset(1.2)
        prof.GetYaxis().SetTitleOffset(1.8)
        prof.Draw()
        txt4 = self.add_desc_to_canvas(0.45, 0.7, 0.85, 0.9, 0.04, True, False, True, False)
        txt4.Draw()
        #cev.cd(5)
        #h_deltas_vs_dist.GetXaxis().SetTitle("Numeric R distorsion (cm)")
        #h_deltas_vs_dist.GetYaxis().SetTitle("(Predicted - Numeric) R distorsion (cm)")
        #h_deltas_vs_dist.Draw("colz")
        cev.SaveAs("%s/canvas_%s_nEv%d.pdf" % (self.dirplots, suffix, self.train_events))

    def plot(self):
        self.logger.info("DnnOptimizer::plot")
        gROOT.ForceStyle()
        for iname, opt in enumerate(self.opt_predout):
            if opt == 1:
                opt_name = self.nameopt_predout[iname]

                myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                                    (self.dirval, self.suffix, self.train_events), "open")
                h_dist_all_events = myfile.Get("%s_all_events_%s" % (self.h_dist_name, self.suffix))
                h_deltas_all_events = myfile.Get("%s_all_events_%s" % \
                                                 (self.h_deltas_name, self.suffix))
                h_deltas_vs_dist_all_events = myfile.Get("%s_all_events_%s" % \
                                                         (self.h_deltas_vs_dist_name, self.suffix))
                profile_deltas_vs_dist_all_events = \
                    myfile.Get("%s_all_events_%s" % (self.profile_name, self.suffix))
                self.plot_distorsion(h_dist_all_events, h_deltas_all_events,
                                     h_deltas_vs_dist_all_events, profile_deltas_vs_dist_all_events,
                                     self.suffix, opt_name)

                counter = 0
                for iexperiment in self.partition['apply']:
                    h_suffix = "Ev%d_Mean%d_%s" % (iexperiment[0], iexperiment[1], self.suffix)
                    h_dist = myfile.Get("%s_%s" % (self.h_dist_name, h_suffix))
                    h_deltas = myfile.Get("%s_%s" % (self.h_deltas_name, h_suffix))
                    h_deltas_vs_dist = myfile.Get("%s_%s" % (self.h_deltas_vs_dist_name, h_suffix))
                    profile = myfile.Get("%s_%s" % (self.profile_name, h_suffix))
                    self.plot_distorsion(h_dist, h_deltas, h_deltas_vs_dist, profile,
                                         h_suffix, opt_name)
                    counter = counter + 1
                    if counter > 100:
                        return


    # pylint: disable=no-self-use
    def gridsearch(self):
        self.logger.warning("Grid search not yet implemented")


    def setup_canvas(self, hist_name, opt_name, x_label, y_label):
        full_name = "%s_canvas_%s_%s" % (hist_name, self.suffix, opt_name)
        canvas = TCanvas(full_name, full_name, 0, 0, 800, 800)
        canvas.SetMargin(0.12, 0.05, 0.12, 0.05)
        canvas.SetTicks(1, 1)

        frame = canvas.DrawFrame(-5, -0.5, +5, +0.5)
        frame.GetXaxis().SetTitle(x_label)
        frame.GetYaxis().SetTitle(y_label)
        frame.GetXaxis().SetTitleOffset(1.5)
        frame.GetYaxis().SetTitleOffset(1.5)
        frame.GetXaxis().CenterTitle(True)
        frame.GetYaxis().CenterTitle(True)
        frame.GetXaxis().SetTitleSize(0.04)
        frame.GetYaxis().SetTitleSize(0.04)
        frame.GetXaxis().SetLabelSize(0.04)
        frame.GetYaxis().SetLabelSize(0.04)

        leg = TLegend(0.5, 0.7, 0.9, 0.9)
        leg.SetBorderSize(0)
        leg.SetTextFont(42)
        leg.SetTextSize(0.03)
        leg.SetHeader("Train setup: #it{N}_{ev}^{training}, #it{n}_{#it{#varphi}}" +\
                      " #times #it{n}_{#it{r}} #times #it{n}_{#it{z}}", "C")

        return canvas, frame, leg


    def save_canvas(self, canvas, frame, prefix, func_name, file_formats):
        file_name = "%s_wide_%s_%s" % (prefix, func_name, self.suffix)
        for file_format in file_formats:
            canvas.SaveAs("%s.%s" % (file_name, file_format))
        frame.GetYaxis().SetRangeUser(-0.05, +0.05)
        file_name = "%s_zoom_%s_%s" % (prefix, func_name, self.suffix)
        for file_format in file_formats:
            canvas.SaveAs("%s.%s" % (file_name, file_format))


    def add_desc_to_canvas(self, xmin, ymin, xmax, ymax, size,
                           add_gran, add_inputs, add_events, add_alice):
        txt1 = TPaveText(xmin, ymin, xmax, ymax, "NDC")
        txt1.SetFillColor(kWhite)
        txt1.SetFillStyle(0)
        txt1.SetBorderSize(0)
        txt1.SetTextAlign(12) # middle,left
        txt1.SetTextFont(42) # helvetica
        txt1.SetTextSize(size)
        if add_alice:
            txt1.AddText("ALICE work in progress")
        if add_gran:
            gran_desc = "#it{n}_{#it{#varphi}} #times #it{n}_{#it{r}} #times #it{n}_{#it{z}}"
            gran_str = "%d #times %d #times %d" % (self.grid_phi, self.grid_r, self.grid_z)
            txt1.AddText("%s = %s" % (gran_desc, gran_str))
        if add_inputs:
            if self.opt_train[0] == 1 and self.opt_train[1] == 1:
                txt1.AddText("inputs: #it{#rho}_{SC} - <#it{#rho}_{SC}>, <#it{#rho}_{SC}>")
            elif self.opt_train[1] == 1:
                txt1.AddText("inputs: #it{#rho}_{SC} - <#it{#rho}_{SC}>")
        if add_events:
            txt1.AddText("#it{N}_{ev}^{training} = %d" % self.train_events)
            # txt1.AddText("#it{N}_{ev}^{validation} = %d" % self.test_events)
            # txt1.AddText("#it{N}_{ev}^{apply} = %d" % self.apply_events)
        txt1.AddText("%d epochs" % self.epochs)
        return txt1


    def draw_multievent_hist(self, events_counts, func_label, hist_name, source_hist):
        gROOT.ForceStyle()
        gran_str = "%d#times %d #times %d" % (self.grid_phi, self.grid_r, self.grid_z)
        date = datetime.date.today().strftime("%Y%m%d")

        file_formats = ["pdf", "png"]
        # file_formats = ["png", "eps", "pdf"]
        var_labels = ["r", "r#varphi", "z"]
        colors = [kBlue+1, kGreen+2, kRed+1, kCyan+2, kOrange+7, kMagenta+2]
        #colors = [kRed+1, kMagenta+2, kOrange+7, kCyan+1, kMagenta+2]
        for iname, opt in enumerate(self.opt_predout):
            if opt == 1:
                opt_name = self.nameopt_predout[iname]
                var_label = var_labels[iname]

                x_label = "d#it{%s}_{true} (cm)" % var_label
                y_label = "%s of d#it{%s}_{pred} - d#it{%s}_{true} (cm)" %\
                          (func_label, var_label, var_label)
                canvas, frame, leg = self.setup_canvas(hist_name, opt_name, x_label, y_label)

                for i, (train_events, _, _, _) in enumerate(events_counts):
                    filename = "%s/output_%s_nEv%d.root" % (self.dirval, self.suffix, train_events)
                    self.logger.info("Reading %s...", filename)

                    root_file = TFile.Open(filename, "read")
                    hist = root_file.Get("%s_all_events_%s" % (source_hist, self.suffix))
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
                                (self.profile_name, self.suffix))
                        hist_stddev = root_file.Get("%s_all_events_%s" % \
                                (self.h_std_dev_name, self.suffix))
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
                txt = self.add_desc_to_canvas(0.15, 0.81, 0.4, 0.89, 0.03,
                                              False, True, False, False)
                txt.Draw()
                self.save_canvas(canvas, frame, "{}/{}".format(self.dirplots, date),
                                 hist_name, file_formats)

    def draw_profile(self, events_counts):
        self.draw_multievent_hist(events_counts, "#it{#mu}", "profile", self.profile_name)

    def draw_std_dev(self, events_counts):
        self.draw_multievent_hist(events_counts, "#it{#sigma}_{std}", "std_dev",
                                  self.h_std_dev_name)

    def draw_mean_std_dev(self, events_counts):
        self.draw_multievent_hist(events_counts, "#it{#mu} #pm #it{#sigma}_{std}", "mean_std_dev",
                self.profile_name)

    def set_ranges(self, ranges, total_events, train_events, test_events, apply_events):
        self.total_events = total_events
        self.train_events = train_events
        self.test_events = test_events
        self.apply_events = apply_events

        self.indices_events_means, self.partition = get_event_mean_indices(
            self.maxrandomfiles, self.range_mean_index, ranges)

        for part in self.partition:
            events_inds = np.array(self.partition[part])
            events_file = "%s/events_%s_%s_nEv%d.csv" % \
                          (self.dirmodel, part, self.suffix, self.train_events)
            np.savetxt(events_file, events_inds, delimiter=",", fmt="%d")

        self.logger.info("Processing %d events", self.total_events)
