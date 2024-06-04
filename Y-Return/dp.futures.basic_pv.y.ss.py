"""SS/FutCndl --> SS/Y/CloseRtn
生成K线数据对应的未来收益率, 并输出到 /var/data/StackData/futures/m{XX}e/Y/CloseRtn for XX in [m01, m05, m15]
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.expanduser("~/futures"))
from utils.logger import log_me
from utils.io.ss import PySharedStack

# PTH_CNDL = "/var/data/StackData/FutCndl"
# PTH_OUT  = "/var/data/StackData/futures/m01e/Y/CloseRtn"

BarLen = [60, 300, 900]
PRED_LEN = [1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120]


DTYPE_Y = np.dtype([("tp", np.int64), ("ticker", "S8")] + [(f"CloseRtn{x:03d}", np.float32) for x in PRED_LEN])


def _get_pth_cndl(ticker: str, barLen):
    _m = {60: 'm01', 300: 'm05', 900: 'm15'}
    return f"/var/data/StackData/futures/{_m[barLen]}e/candle/{ticker[:-4]}/{ticker}.ss"


def _get_pth_out(ticker: str, barLen):
    _m = {60: 'm01', 300: 'm05', 900: 'm15'}
    return f"/var/data/StackData/futures/{_m[barLen]}e/Y/CloseRtn/{ticker[:-4]}/{ticker}.ss"


def _divide(dividend, divisor):
    if isinstance(divisor, pd.Series) or isinstance(divisor, pd.DataFrame):
        return dividend / divisor.where(divisor != 0)
    elif isinstance(divisor, np.ndarray):
        return dividend / np.where(divisor == 0, np.nan, divisor)
    else:
        raise TypeError("该除法只适用于除数为numpy, pandas或xarray数据格式")


# @log_me
def update_ticker(ticker:str):
    """update SS/Y/CloseRtn for single ticker
    """
    # print(f"updating SS/Y/CloseRtn for {ticker}")
    # ticker = 'rb2310'
    for _bar in BarLen:
        
        _pth_cndl = _get_pth_cndl(ticker, _bar)

        ss = PySharedStack.PySharedStack(_pth_cndl)

        _pth_out = _get_pth_out(ticker, _bar)

        _pred = max(PRED_LEN)

        idx_read = 0
        if os.path.exists(_pth_out):
            _ss_out_readonly = PySharedStack.PySharedStack(_pth_out)
            idx_read = _ss_out_readonly.size()
            del _ss_out_readonly

        if idx_read >= ss.size() - _pred:
            print(f"SS/Y/CloseRtn already updated.")
            continue

        cndl = ss.read(idx_read, ss.size())

        df = pd.DataFrame(data={'preclose': cndl['preclose'], 'close': cndl['close']})
        df['CloseRtn'] = _divide(df['close'], df['preclose']) - 1
        for k in PRED_LEN:
            df[f'CloseRtn{k:03d}'] = df['CloseRtn'].fillna(0).rolling(k, min_periods=1).sum().shift(-k)

        n_valid = df.shape[0] - _pred
        df = df.iloc[:n_valid, :]

        if 0 < n_valid:
            arr_out = np.empty((n_valid,), dtype=DTYPE_Y)
            arr_out['tp'] = cndl[:n_valid]['UpdateTime']
            arr_out['ticker'] = cndl[:n_valid]['ticker']
            for k in PRED_LEN:
                field = f'CloseRtn{k:03d}'
                arr_out[field] = df[field]

            ss_out = PySharedStack.PySharedStack(_pth_out, DTYPE_Y, True)
            idx_ins = 0
            if ss_out.size() > 1:
                last = ss_out.read(ss_out.size()-1, ss_out.size())
                sz = n_valid
                while idx_ins != sz and arr_out[idx_ins]['tp'] <= last['tp']:
                    idx_ins = idx_ins + 1
            if idx_ins < arr_out.shape[0]:
                ss_out.push_back(arr_out[idx_ins:])
            del ss_out

    # print(f"updating SS/Y/CloseRtn for {ticker}")
    
_pth_cndl_m01_root = "/var/data/StackData/futures/m01e/candle"
all_ss_files = [os.listdir(f"{_pth_cndl_m01_root}/{instrType}") for instrType in os.listdir(_pth_cndl_m01_root)]
all_tickers = list({file[:-3] for sublist in all_ss_files for file in sublist}) # use set comprehension to get unique list of tickers for ss_files

import tqdm
import datetime
from multiprocessing import Pool
with Pool(40) as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(update_ticker, all_tickers), total=len(all_tickers)):
        pass

## debug

# update_ticker('SR1201')