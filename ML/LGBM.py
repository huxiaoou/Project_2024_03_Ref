import lightgbm
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import Subset
import numpy as np
import os
import joblib

class LGBM:

    def __init__(self, cfg_model:dict={}):
        self._cfg = cfg_model
        self.n_features = cfg_model['n_features']
        # set mse as obj
        self.fobj = None
        self.feval = None
        self.params={"objective": "mse", "verbosity": -1, "colsample_bytree": 0.7, 
                     "subsample": 0.7, 'num_leaves': 512, 'learning_rate': 0.1 }
        # self.params={"objective": "mse", "verbosity": -1, 'max_depth': 64 }

    def fit(self, dataset:Dataset):
        _valid_split = self._cfg['Train']['ValidationSplit']
        _len_ds = len(dataset)
        _len_valid = int(_len_ds * _valid_split)
        _idx_valid = list(range(0, _len_valid))
        _idx_train = list(range(_len_valid, _len_ds))
        ds_valid = Subset(dataset, _idx_valid)
        ds_train = Subset(dataset, _idx_train)
        # [ds_train, ds_valid] = random_split(dataset, [_len_ds - _len_valid, _len_valid])
        X_np = np.empty([len(ds_train), self.n_features])
        y_np = np.empty(len(ds_train))
        for _r in range(len(ds_train)):
            _X, _y = ds_train[_r]
            X_np[_r, :] = _X[:]
            y_np[_r] = _y[0]
        X_valid_np = np.empty([len(ds_valid), self.n_features])
        y_valid_np = np.empty(len(ds_valid))
        for _r in range(len(ds_valid)):
            _X, _y = ds_train[_r]
            X_valid_np[_r, :] = _X[:]
            y_valid_np[_r] = _y[0]

        init_loss_train = np.mean(y_np ** 2)
        print(init_loss_train)
        # return
        
        # ds_train_lgb = lightgbm.Dataset(X_np, label=y_np, init_score=X_np.mean(axis=1))# https://datascience.stackexchange.com/questions/17074/what-is-init-score-in-lightgbm
        # ds_valid_lgb = lightgbm.Dataset(X_valid_np, label=y_valid_np, init_score=X_valid_np.mean(axis=1))
        ds_train_lgb = lightgbm.Dataset(X_np, label=y_np)# https://datascience.stackexchange.com/questions/17074/what-is-init-score-in-lightgbm
        ds_valid_lgb = lightgbm.Dataset(X_valid_np, label=y_valid_np)
        evals_result = {}
        callbacks = [lightgbm.log_evaluation(period=20), lightgbm.early_stopping(stopping_rounds=self._cfg['Train']['Stop']['OnLossPlateau'])]
        self.model = lightgbm.train(
            params=self.params,
            train_set=ds_train_lgb,
            num_boost_round=self._cfg['Train']['Stop']['OnMaxEpoch'],
            valid_sets=[ds_train_lgb, ds_valid_lgb],
            valid_names=['train', 'valid'],
            keep_training_booster=True,
            feval=self.feval,
            callbacks=callbacks)
        self.save()
        # import pandas as pd
        # _df_importance = pd.DataFrame({'i_features': list(range(self.n_features)), 'importance': self.model.feature_importance(importance_type='split')})
        # print(_df_importance.sort_values(by='importance', ascending=False).iloc[:10, :])
        # import matplotlib.pyplot as plt
        # lightgbm.plot_importance(self.model)
        # plt.savefig('/home/wangx/futures/models/talib/train_models/importance.png')
        # self.evals_result["train"] = list(evals_result["train"].values())[0]
        # self.evals_result["valid"] = list(evals_result["valid"].values())[0]

    def predict(self, dataset:Dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        X_test = np.empty([len(dataset), self.n_features])
        for _r in range(len(dataset)):
            _X, _y = dataset[_r]
            X_test[_r, :] = _X[:]
        # y_pred = X_test.mean(axis=1) + self.model.predict(X_test)
        y_pred = self.model.predict(X_test)
        return y_pred

    def save(self):
        os.makedirs(self._cfg['ModelSaveFullPath'], exist_ok=True)
        joblib.dump(self.model, f"{self._cfg['ModelSaveFullPath']}/lgbm.pkl")
        
    def load(self):
        self.model = joblib.load(f"{self._cfg['ModelSaveFullPath']}/lgbm.pkl")
