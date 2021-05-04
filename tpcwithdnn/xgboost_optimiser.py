# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
from timeit import default_timer as timer

import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRFRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from root_numpy import fill_hist # pylint: disable=import-error
from ROOT import TFile # pylint: disable=import-error, no-name-in-module

import tpcwithdnn.plot_utils as plot_utils
from tpcwithdnn.debug_utils import log_time
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_train_apply_idc

class XGBoostOptimiser(Optimiser):
    name = "xgboost"

    def __init__(self, config):
        super().__init__(config)
        self.config.logger.info("XGBoostOptimiser::Init")
        self.model = XGBRFRegressor(verbosity=1, **(self.config.params))

    def train(self):
        self.config.logger.info("XGBoostOptimiser::train")
        inputs, exp_outputs = self.get_train_apply_data_("train")
        start = timer()
        self.model.fit(inputs, exp_outputs)
        end = timer()
        log_time(start, end, "actual train")
        if self.config.plot_train:
            start = timer()
            self.plot_train_(inputs, exp_outputs)
            end = timer()
            log_time(start, end, "train plot")
        self.save_model_(self.model)

    def apply(self):
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        self.load_model_()
        inputs, exp_outputs = self.get_train_apply_data_("apply")
        start = timer()
        pred_outputs = self.model.predict(inputs)
        end = timer()
        log_time(start, end, "actual predict")
        start = timer()
        self.plot_apply_(exp_outputs, pred_outputs)
        end = timer()
        log_time(start, end, "plot apply")
        self.config.logger.info("Done apply")

    def search_grid(self):
        raise NotImplementedError("Search grid method not implemented yet")

    def save_model_(self, model):
        # Snapshot - can be used for further training
        out_filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        pickle.dump(model, open(out_filename, 'wb'), protocol=4)

    def load_model_(self):
        # Loading a snapshot
        filename = "%s/xgbmodel_%s_nEv%d.json" %\
                (self.config.dirmodel, self.config.suffix, self.config.train_events)
        self.model = pickle.load(open(filename, 'rb'))

    def get_train_apply_data_(self, partition):
        downsample = self.config.downsample # if partition == "train" else False
        inputs = []
        exp_outputs = []
        for indexev in self.config.partition[partition]:
            inputs_single, exp_outputs_single = load_train_apply_idc(self.config.dirinput_train,
                                                       indexev, self.config.input_z_range,
                                                       self.config.output_z_range,
                                                       self.config.opt_predout,
                                                       downsample, self.config.downsample_frac)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
        inputs = np.concatenate(inputs)
        exp_outputs = np.concatenate(exp_outputs)
        return inputs, exp_outputs

    def plot_apply_(self, exp_outputs, pred_outputs):
        myfile = TFile.Open("%s/output_%s_nEv%d.root" % \
                            (self.config.dirval, self.config.suffix, self.config.train_events),
                            "recreate")
        h_dist_all_events, h_deltas_all_events, h_deltas_vs_dist_all_events =\
                plot_utils.create_apply_histos(self.config, self.config.suffix, infix="all_events_")
        distortion_numeric_flat_m, distortion_predict_flat_m, deltas_flat_a, deltas_flat_m =\
            plot_utils.get_apply_results_single_event(pred_outputs, exp_outputs)

        fill_hist(h_dist_all_events, np.concatenate((distortion_numeric_flat_m, \
                                                     distortion_predict_flat_m), axis=1))
        fill_hist(h_deltas_all_events, deltas_flat_a)
        fill_hist(h_deltas_vs_dist_all_events,
                  np.concatenate((distortion_numeric_flat_m, deltas_flat_m), axis=1))

        h_dist_all_events.Write()
        h_deltas_all_events.Write()
        h_deltas_vs_dist_all_events.Write()
        prof_all_events = h_deltas_vs_dist_all_events.ProfileX()
        prof_all_events.SetName("%s_all_events_%s" % (self.config.profile_name,
                                                      self.config.suffix))
        prof_all_events.Write()
        plot_utils.fill_std_dev_apply_hist(h_deltas_vs_dist_all_events, self.config.h_std_dev_name,
                                           self.config.suffix, "all_events_")

        myfile.Close()

    def plot_train_(self, x_data, y_data):
        plt.figure()
        #plt.yscale("log")
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2)
        train_errors, val_errors = [], []
        high = len(x_train)
        low = 0
        step = int((high - low) / self.config.train_plot_npoints)
        checkpoints = np.arange(start=step, stop=high+1, step=step)
        for checkpoint in checkpoints:
            self.model.fit(x_train[:checkpoint], y_train[:checkpoint])
            y_train_predict = self.model.predict(x_train[:checkpoint])
            y_val_predict = self.model.predict(x_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:checkpoint]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
        plt.plot(checkpoints, np.sqrt(train_errors), ".", label="train")
        plt.plot(checkpoints, np.sqrt(val_errors), ".", label="validation")
        plt.ylim([0, np.amax(np.sqrt(val_errors)) * 2])
        plt.title("Learning curve BDT")
        plt.xlabel("Training set size")
        plt.ylabel("RMSE")
        plt.legend(loc="lower left")
        plt.savefig("%s/learning_plot_%s_nEv%d.png" % (self.config.dirplots, self.config.suffix,
                                                       self.config.train_events))
