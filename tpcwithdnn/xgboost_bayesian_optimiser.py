# pylint: disable=missing-module-docstring, missing-function-docstring, missing-class-docstring
# pylint: disable=too-many-instance-attributes
import json
import pickle
from xgboost import XGBRFRegressor

from hyperopt import hp
from hyperopt.pyll import scope

from machine_learning_hep.optimisation.bayesian_opt import BayesianOpt
from machine_learning_hep.optimisation.metrics import get_scorers

from tpcwithdnn.logger import get_logger

class XGBoostBayesianOptimiser(BayesianOpt):
    space = {"n_estimators": scope.int(hp.quniform("x_n_estimators", 100, 500, 10)),
             "max_depth": scope.int(hp.quniform("x_max_depth", 1, 12, 1)),
             "learning_rate": hp.uniform("x_learning_rate", 0.0005, 1.5),
             "gamma": hp.uniform("x_gamma", 0.0, 0.2),
             "min_child_weight": scope.int(hp.quniform("x_min_child", 1, 10, 1)),
             "subsample": hp.uniform("x_subsample", 0.5, 0.9),
             "colsample_bytree": hp.uniform("x_colsample_bytree", 0.5, 1.),
             "colsample_bylevel": hp.uniform("x_colsample_bylevel", 0.5, 1.),
             "colsample_bynode": hp.uniform("x_colsample_bynode", 0.5, 1.),
             "reg_alpha": hp.uniform("x_reg_alpha", 0, 1),
             "reg_lambda": hp.uniform("x_reg_lambda", 0, 1),
             "base_score": hp.uniform("x_base_score", 0.2, 0.8),
             "scale_pos_weight": scope.int(hp.quniform("x_scale_pos_weight", 1, 10, 1))}

    # max delta step = 0 means no constraints
    # n_jobs = -1 use all cores
    config = {"n_gpus": 0, "n_jobs": 2, "tree_method": "hist", "max_delta_step": 0}

    def __init__(self):
        super().__init__(self.config, self.space)
        self.logger = get_logger()
        self.logger.info("XGBoostBayesianOptimiser::Init")
        self.nkfolds = 3
        self.scoring = get_scorers(["MSE"])
        self.scoring_opt = "MSE"
        self.low_is_better = True
        self.n_trials = 100
        self.score_train_test_diff = 0.00001
        self.ncores = 3

    def yield_model_(self, model_config, space): # pylint: disable=unused-argument, no-self-use
        config = self.next_params(space)
        return XGBRFRegressor(verbosity=1, **(config)), config

    def save_model_(self, model, out_dir): # pylint: disable=no-self-use
        with open("%s/bayes_xgbmodel.json" % out_dir, "wb") as model_file:
            pickle.dump(model, model_file, protocol=4)
