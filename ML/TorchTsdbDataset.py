import pytsdb
import torch
from torch.utils.data import Dataset
from torch import tensor
import numpy as np
import pandas as pd

class TorchTsdbDataset(Dataset):

    def __init__(self, db:str, tbl: str, X_cols:list, y_col:str, beg: str, end: str, hot_only:bool=False, filter=np.nan):
        _beg = pytsdb.parse_time(beg)
        _end = pytsdb.parse_time(end)
        _db = pytsdb.PyTsdb(db)
        # _tbl_x = tbl[:3] + "b"
        _tbl_x = tbl
        all_cols = X_cols + [y_col] if len(y_col)>0 else X_cols
        for col in all_cols:
            col_end = _db.next_ts_str(_tbl_x, col)
            if col_end < _end:
                print(f"end adjusted to {col_end}. old end: {_end}, reason: {_tbl_x}.{col} ends before {col_end}")
                _end = col_end
        _data = _db.read_columns(_tbl_x, all_cols, _beg, _end)
        _status = _db.read_columns(_tbl_x, ['status'], _beg, _end, dtypes={'status': np.int32})
        assert(_status.shape[0] == _data.shape[0])
        df_status = pd.DataFrame(_status)
        df_status['idx'] = np.arange(df_status.shape[0])
        if hot_only:
            hot = _db.read_columns("d01b", ['hot'], _beg, _end, dtypes={'hot': np.int32})
            df_hot = pd.DataFrame(hot)
            df_status = pd.merge(df_status, df_hot[['ii', 'trading_day', 'hot']], on=['ii', 'trading_day'], how='left')
            df_status = df_status.loc[df_status['hot']!=0, :]
        df_status = df_status.loc[df_status['status']!=0, :]
        _data = _data[df_status['idx']]
        X = np.empty([_data.shape[0], len(X_cols)], dtype=np.float32)
        for iCol in range(len(X_cols)):
            X[:,iCol] = _data[X_cols[iCol]]
        yy = np.empty([_data.shape[0], 1], dtype=np.float32)
        if 0 < len(y_col):
            yy[:,0] = _data[y_col]
        _na = (np.isnan(X).any(axis=1) | np.isnan(yy).any(axis=1)) if len(y_col) > 0 else np.isnan(X).any(axis=1)
        if filter > 0:
            _na = _na | ((yy < filter) & (yy > -filter)).any(axis=1)
        X = X[~_na, :]
        yy = yy[~_na]
        _data = _data[~_na]
        self._X = X
        self._y = yy
        self._tsdb_index = pd.DataFrame({"tp": _data['tp'], "ii": _data['ii']})

        
    def __len__(self):
        return self._X.shape[0]


    def __getitem__(self, idx):
        return self._X[idx, :], self._y[idx]
    

    def get_tsdb_index(self):
        return self._tsdb_index


    # def __getitems__():
    #     pass
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def test():
    ds = TorchTsdbDataset("/var/TSDB/futures", "d01b", 
                          ['bpreclose', 'bpreoi'], 
                          'Y.CloseRtn05', 
                          '20120101 20:00:00.000', 
                          '20231215 17:00:00.000', 
                          True)
    dl_train = DataLoader(ds, batch_size=1000)
    for X, y in dl_train:
        print(f"X {X.shape[0]}-by-{X.shape[1]}, y {y.shape[0]}-by-{y.shape[1]}")

    [ds_train, ds_valid] = random_split(ds, [0.7, 0.3])

    print(len(ds_train))
    print(len(ds_valid))

if __name__ == "__main__":
    test()