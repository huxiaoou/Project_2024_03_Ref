import importlib
import sys

def create_model(model_cfg:dict):
    if model_cfg['Module']['path'] not in sys.path:
        sys.path.append(model_cfg['Module']['path'])
    m = importlib.import_module(model_cfg['Module']['package'] if 'package' in model_cfg['Module'] else model_cfg['Module']['class'])
    return getattr(m, model_cfg['Module']['class'])(model_cfg)
    