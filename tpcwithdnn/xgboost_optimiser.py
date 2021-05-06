# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
from timeit import default_timer as timer

import pickle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRFRegressor

from sklearn.metrics import mean_squared_error

from root_numpy import fill_hist # pylint: disable=import-error
from ROOT import TFile # pylint: disable=import-error, no-name-in-module

import tpcwithdnn.plot_utils as plot_utils
from tpcwithdnn.debug_utils import log_time, log_memory_usage, log_total_memory_usage
from tpcwithdnn.optimiser import Optimiser
from tpcwithdnn.data_loader import load_event_idc

class XGBoostOptimiser(Optimiser):
    name = "xgboost"

    def __init__(self, config):
        super().__init__(config)
        self.config.logger.info("XGBoostOptimiser::Init")
        self.model = XGBRFRegressor(verbosity=1, **(self.config.params))

    def train(self):
        self.config.logger.info("XGBoostOptimiser::train")
        start = timer()
        inputs, exp_outputs = self.get_data_("train")
        end = timer()
        log_time(start, end, "load train")
        log_memory_usage(((inputs, "Input train data"), (exp_outputs, "Output train data")))
        self.config.logger.info("Memory usage after loading data")
        log_total_memory_usage()
        if self.config.plot_train:
            inputs_val, outputs_val = self.get_data_("validation")
            log_memory_usage(((inputs_val, "Input val data"), (outputs_val, "Output val data")))
            self.config.logger.info("Memory usage after loading val data")
            log_total_memory_usage()
            start = timer()
            self.plot_train_(inputs, exp_outputs, inputs_val, outputs_val)
            end = timer()
            log_time(start, end, "train plot")
        start = timer()
        self.model.fit(inputs, exp_outputs)
        end = timer()
        log_time(start, end, "actual train")
        self.save_model_(self.model)

    def apply(self):
        self.config.logger.info("XGBoostOptimiser::apply, input size: %d", self.config.dim_input)
        self.load_model_()
        start = timer()
        inputs, exp_outputs = self.get_data_("apply")
        end = timer()
        log_time(start, end, "load apply")
        log_memory_usage(((inputs, "Input apply data"), (exp_outputs, "Output apply data")))
        self.config.logger.info("Memory usage after loading apply data")
        log_total_memory_usage()
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

    def get_data_(self, partition):
        downsample = self.config.downsample if partition == "train" else False
        inputs = []
        exp_outputs = []
        for ind, indexev in enumerate(self.config.partition[partition]):
            inputs_single, exp_outputs_single = load_event_idc(self.config.dirinput_train,
                                                               indexev, self.config.input_z_range,
                                                               self.config.output_z_range,
                                                               self.config.opt_predout,
                                                               downsample,
                                                               self.config.downsample_frac)
            inputs.append(inputs_single)
            exp_outputs.append(exp_outputs_single)
            log_memory_usage(((inputs, "%d input data" % ind),
                              (exp_outputs, "%d output data" % ind)))
            log_memory_usage(((inputs_single, "%d single input data" % ind),
                              (exp_outputs_single, "%d single output data" % ind)))
            self.config.logger.info("Memory usage after loading data %d" % ind)
            log_total_memory_usage()
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

    def plot_train_(self, x_train, y_train, x_val, y_val):
        plt.figure()
        #plt.yscale("log")
        train_errors, val_errors = [], []
        data_size = len(x_train)
        size_per_event = int(data_size / self.config.train_events)
        step = int(data_size / self.config.train_plot_npoints)
        checkpoints = np.arange(start=size_per_event, stop=data_size, step=step)
        for ind, checkpoint in enumerate(checkpoints):
            self.model.fit(x_train[:checkpoint], y_train[:checkpoint])
            y_train_predict = self.model.predict(x_train[:checkpoint])
            y_val_predict = self.model.predict(x_val)
            train_errors.append(mean_squared_error(y_train_predict, y_train[:checkpoint]))
            val_errors.append(mean_squared_error(y_val_predict, y_val))
            if ind in (0, self.config.train_plot_npoints // 2, self.config.train_plot_npoints - 1):
                self.plot_results_(y_train[:checkpoint], y_train_predict, "train-%d" % ind)
                self.plot_results_(y_val, y_val_predict, "val-%d" % ind)
        self.config.logger.info("Memory usage during plot train")
        log_total_memory_usage()
        plt.plot(checkpoints, np.sqrt(train_errors), ".", label="train")
        plt.plot(checkpoints, np.sqrt(val_errors), ".", label="validation")
        plt.ylim([0, np.amax(np.sqrt(val_errors)) * 2])
        plt.title("Learning curve BDT")
        plt.xlabel("Training set size")
        plt.ylabel("RMSE")
        plt.legend(loc="lower left")
        plt.savefig("%s/learning_plot_%s_nEv%d.png" % (self.config.dirplots, self.config.suffix,
                                                       self.config.train_events))
        plt.clf()

    def plot_results_(self, exp_outputs, pred_outputs, infix):
        plt.figure()
        plt.plot(exp_outputs, pred_outputs, ".")
        plt.xlabel("Expected output")
        plt.ylabel("Predicted output")
        plt.savefig("%s/num-exp-%s_%s_nEv%d.png" % (self.config.dirplots, infix, self.config.suffix,
                                                   self.config.train_events))
        plt.clf()
