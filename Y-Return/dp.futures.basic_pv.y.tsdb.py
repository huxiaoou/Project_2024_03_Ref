"""SS/Y/CloseRtn --> TSDB/Y/CloseRtn
1. 通过调用 qalign 将 /var/data/StackData/futures/m01e/Y/CloseRtn 中的数据对齐到 /var/TSDB/futures/ m01e.Y.CloseRtn 中
2. 通过 qdata subset_upd 生成 /var/TSDB/FutHot 对应的 m01e.Y.CloseRtn 数据
# 2. CloseRtn / d01e.volatility 存储到 m01e.Y.VolatilityAdjRtn 中
"""

import os, sys
import pandas as pd
import numpy as np

sys.path.append(os.path.expanduser("~/futures"))
from utils.logger import log_me
from utils.io.ss import PySharedStack

FREQs = ['m01', 'm05', 'm15']


for freq in FREQs:

    PTH_Y_SS  = f"/var/data/StackData/futures/{freq}e"
    TP_END = 'TradingDay-2 17:00:00.000'

    ss = PySharedStack.PySharedStack(f"{PTH_Y_SS}/Y/CloseRtn/rb/rb2310.ss")
    sample = ss.read(0, 1)

    fields = list(sample.dtype.names)
    fields = [x for x in fields if x not in ['tp', 'uid', 'ticker', 'trading_day', 'barno']]
    fields_cmd = ' --fields '.join(fields)

    qalign_cmd = f"qalign -v update --ncpu 50 --end '{TP_END}' --ignore-missing-ticker --path {PTH_Y_SS} --db /var/TSDB/futures --tbl {freq}e --exact --float32 --prefix Y.CloseRtn --fields {fields_cmd}"
    print(qalign_cmd)
    os.system(qalign_cmd)

    #===========================================================================================================
    #region "将数据抽取到 /var/TSDB/FutHot 中"
    #===========================================================================================================

    cmd_subset_to_hot = f"qdata subset_upd --db /var/TSDB/FutHot --idx hot --from /var/TSDB/futures --tbl {freq}e --prefix Y.CloseRtn --path"
    print(cmd_subset_to_hot)
    os.system(cmd_subset_to_hot)

    #===========================================================================================================
    #endregion "将数据抽取到 /var/TSDB/FutHot 中"
    #===========================================================================================================