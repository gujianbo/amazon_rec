import xgboost
import os
import sys
sys.path.append("..")
from utils.args import config
from tqdm.auto import tqdm
import hashlib
import logging
import numpy as np
import pandas as pd
import glob

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d]- %(message)s"
logging.basicConfig(filename=config.log_file, level=logging.DEBUG, format=LOG_FORMAT)


class Iterator(xgboost.DataIter):
    """A custom iterator for loading files in batches."""
    def __init__(self, file_paths):
        self._file_paths = file_paths
        self._it = 0
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_file(self):
        data_file = self._file_paths[self._it]
        train_x_arr = []
        train_y_arr = []
        with open(data_file, "r") as fd:
            for line in tqdm(fd, desc="load train"):
                line = line.strip()
                (prev_items, candi, locale_code, item_feat_str, session_stat_feat_str, interact_feat_str,
                 label) = line.split("\t")
                feat = []
                feat += [float(locale_code)]
                feat += [float(item) for item in item_feat_str.split(",")]
                feat += [float(item) for item in session_stat_feat_str.split(",")]
                feat += [float(item) for item in interact_feat_str.split(",")]
                train_x_arr.append(feat)
                train_y_arr.append([float(label)])
        X = np.array(train_x_arr)
        y = np.array(train_y_arr)
        logging.info(f"load file {data_file} train_x.shape:{X.shape}")
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``
        """
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the similar signature to
        # the ``DMatrix`` constructor.
        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return 1

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0


def train_gbm(train_file, test_file, model_file):
    train_files = [train_file+".1.1", train_file+".1.2", train_file+".1.3", train_file+".1.4", train_file+".1.5",
                   train_file+".2.1", train_file+".2.2", train_file+".2.3", train_file+".2.4", train_file+".2.5",
                   train_file+".3.1", train_file+".3.2", train_file+".3.3", train_file+".3.4", train_file+".3.5"]
    test_files = [test_file+".1.1", test_file+".2.1", test_file+".3.1"]
    logging.info(f"train_files:{train_files}")
    logging.info(f"test_files:{test_files}")
    train_it = Iterator(train_files)
    test_it = Iterator(test_files)

    missing = np.NaN
    train_data = xgboost.DMatrix(train_it, missing=missing, enable_categorical=False)
    test_data = xgboost.DMatrix(test_it, missing=missing, enable_categorical=False)

    param = {
        'learning_rate': 0.1,
        'max_depth': 6,
        # 'num_trees': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'subsample': 0.7,
        'scale_pos_weight': config.scale_pos_weight,
        'colsample_bytree': 0.8
    }
    model = xgboost.train(param,
                      dtrain=train_data,
                      evals=[(train_data, 'train'), (test_data, 'valid')],
                      num_boost_round=1000,
                      early_stopping_rounds=200,
                      verbose_eval=10)
    model.save_model(model_file)

    dd = model.get_score(importance_type='weight')
    importances_weight = pd.DataFrame({'feature': dd.keys(), f'importance_weight': dd.values()})
    dd = model.get_score(importance_type='gain')
    importances_gain = pd.DataFrame({'feature': dd.keys(), f'importance_gain': dd.values()})
    importances_weight.to_csv(model_file+".weight")
    importances_gain.to_csv(model_file+".gain")

    logging.info(f"finish!")


if __name__ == "__main__":
    logging.info(f"train_file:{config.train_file}")
    logging.info(f"test_file:{config.test_file}")
    logging.info(f"model_file:{config.model_file}")
    train_gbm(config.train_file, config.test_file, config.model_file)
