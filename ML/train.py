import yaml
import pytsdb
import os
import pandas as pd
import numpy as np
from .model_factory import create_model
from .TorchTsdbDataset import TorchTsdbDataset


class Trainer:
    
    def __init__(self, pth_cfg):
        with open(pth_cfg, 'r') as f:
            self.CFG = yaml.safe_load(f)
        self._parse_info_from_config()
        
    def _parse_info_from_config(self):

        CFG = self.CFG 

        db = pytsdb.PyTsdb(CFG['DataSet']['DB'])
        self.TBL = CFG['DataSet']['TBL']

        self.FEATURES = CFG['DataSet']['X']
        CFG['Model']['n_features'] = len(self.FEATURES)
        self.Y = CFG['DataSet']['y']

        _tp_end_data = [db.next_ts_str(self.TBL, x) for x in self.FEATURES + [self.Y]]
        print("\n\nstatus of input data: ")
        print(dict(zip(self.FEATURES+[self.Y], _tp_end_data)))
        self.TP_END_DATA = min(_tp_end_data)

        self.TP_BEG_DATA = db.next_ts_str(self.TBL, "__column_not_existing_20231222dsgasewrwrewtr2gsag_") # 一个不存在的数据，查到 TBL的起点
        if self.TP_BEG_DATA < f"{CFG['DataSet']['BEG']} 07:00:00.000":
            self.TP_BEG_DATA = f"{CFG['DataSet']['BEG']} 07:00:00.000"

        self.PTH_MODEL_SAVE = CFG['Model']["Save"]["Root"] + "/" + CFG['ID']
        os.makedirs(self.PTH_MODEL_SAVE, exist_ok=True)

        _info = db.read_columns("d01b", [], self.TP_BEG_DATA, "TradingDay+1 10:00:00.000")
        TradingDays = np.unique(_info['trading_day'])
        TradingDays = TradingDays[TradingDays>0]
        TradingDays = pd.DataFrame({'daily': TradingDays})
        TradingDays['yearly'] = (TradingDays['daily'] / 10000).astype(int)
        TradingDays['monthly'] = (TradingDays['daily'] / 100).astype(int)

        TP_TRAINs = TradingDays.groupby(CFG['Model']['Train']['Step']).agg(FIRST=('daily', min), LAST=('daily', max))
        TP_TRAINs['BEG'] = TP_TRAINs['FIRST'].shift(CFG['Model']['Train']['RollingWindow']['MAX']-1).fillna(TP_TRAINs.iloc[0, :]['FIRST']).astype(int)
        TP_TRAINs = TP_TRAINs.iloc[CFG['Model']['Train']['RollingWindow']['MIN']-1:-1,]
        if TP_TRAINs.iloc[-1,:]['LAST'] > int(self.TP_END_DATA[:8]):
            TP_TRAINs = TP_TRAINs.iloc[:-1, :]
        
        self.TP_TRAINs = TP_TRAINs.loc[:, ['BEG', 'LAST']]
        #              BEG      LAST
        # yearly                    
        # 2012    20120105  20121231
        # 2013    20120105  20131231
        # 2014    20120105  20141231                                                                                                                                                                                                                                                                                                                       
        # 2015    20130104  20151231
        # 2016    20140102  20161230
    
    def train(self):
                
        CFG = self.CFG

        for train_intervals in self.TP_TRAINs.itertuples():
            train_beg = train_intervals[1]
            train_last = train_intervals[2]

            pth_mdl = f"{self.PTH_MODEL_SAVE}/{train_last}/FROM_{train_beg}"
            if os.path.exists(pth_mdl):
                print(f"{pth_mdl} already exist. maybe [{train_beg}, {train_last}] is already trained. remove this folder if you want to train again.")
                continue
            print(
            f'''
            ============================================================================================================
            TRAIN {train_beg}-{train_last}: Loading Data
            ============================================================================================================
            '''
            )
            dataset = TorchTsdbDataset(CFG['DataSet']['DB'],
                                    CFG['DataSet']['TBL'],
                                    CFG['DataSet']['X'],
                                    CFG['DataSet']['y'],
                                    f"{train_beg} 07:00:00.000",
                                    f"{train_last} 17:00:00.000",
                                    (CFG['DataSet']).get('hot_only', False),
                                    (CFG['DataSet']).get('filter', np.nan))
            print(
            f'''
            ============================================================================================================
            TRAIN {train_beg}-{train_last}: Training
            ============================================================================================================
            '''
            )
            cfg_model = CFG['Model']
            cfg_model["ModelSaveFullPath"] = pth_mdl
            mdl = create_model(cfg_model)
            mdl.fit(dataset)

    
    def predict(self):
        CFG = self.CFG
        TBL = self.TBL
        FEATURES = self.FEATURES
        TP_TRAINs = self.TP_TRAINs
        PTH_MODEL_SAVE = self.PTH_MODEL_SAVE
        db = pytsdb.PyTsdb(CFG['DataSet']['DB'])
        db_pred = pytsdb.PyTsdb(CFG["Predict"]["DB"])
        col_pred = CFG["Predict"]["Col"]
        TP_BEG_PREDICT = db_pred.next_ts_str(TBL, col_pred)
        _tp_end_features = min([db.next_ts_str(TBL, x) for x in FEATURES])
        cfg_model = CFG['Model']
        while TP_BEG_PREDICT < _tp_end_features:
            k = 0
            while k < TP_TRAINs.shape[0] and not (TP_BEG_PREDICT < f"{TP_TRAINs.iloc[k, :]['LAST']} 20:01:00.000"):
                k = k + 1
            if 0 == k:# 样本内预测值设置为 nan
                _pred = pd.DataFrame(db_pred.read_columns(TBL, [], TP_BEG_PREDICT
                                                          , f"{TP_TRAINs.iloc[0, :]['LAST']} 17:00:00.000"))
                _pred[col_pred] = np.nan
                db_pred.insert_column(TBL, col_pred, _pred)
            else:
                cfg_model['ModelSaveFullPath'] = f"{PTH_MODEL_SAVE}/{TP_TRAINs.iloc[k-1, :]['LAST']}/FROM_{TP_TRAINs.iloc[k-1, :]['BEG']}" #截止上一个训练周期的模型用于样本外预测
                pred_beg = f"{TP_TRAINs.iloc[k-1, :]['LAST']} 19:00:00.000"
                pred_end = f"{TP_TRAINs.iloc[k, :]['LAST']} 17:00:00.000" if k != TP_TRAINs.shape[0] else _tp_end_features
                dataset = TorchTsdbDataset(CFG['DataSet']['DB'],
                                    CFG['DataSet']['TBL'],
                                    CFG['DataSet']['X'],
                                    "",# empty y
                                    pred_beg,
                                    pred_end,
                                    CFG['DataSet']['hot_only'])
                mdl = create_model(cfg_model)
                mdl.load()
                y_pred = mdl.predict(dataset)
                df_pred = dataset.get_tsdb_index().copy()
                df_pred[col_pred] = y_pred
                df_all = pd.DataFrame(db_pred.read_columns(TBL, [], pred_beg, pred_end))
                df_insert = pd.merge(df_all.loc[:, ['tp', 'ii']], df_pred.loc[:, ['tp', 'ii', col_pred]], on=['tp', 'ii'], how='left')
                db_pred.insert_column(TBL, col_pred, df_insert)
            TP_BEG_PREDICT = db_pred.next_ts_str(CFG["DataSet"]["TBL"], col_pred)

    def status(self):
        CFG = self.CFG
        os.system(f"qdata stat --db {CFG['Predict']['DB']} --tbl {self.TBL} --nonzero --tail 20 --col {CFG['Predict']['Col']}")
