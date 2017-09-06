
import mxnet as mx
from utils.load_model import load_param
from module import MutableModule

class Predictor(object):
    def __init__(self, symbol, prefix, epoch, provide_data, provide_label=[], ctx=mx.cpu(), arg_params=None, aux_params=None, has_json_symbol=False):
        if has_json_symbol:
            symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        elif arg_params is None:
            arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
        
        self.symbol = symbol
        data_names = [k[0] for k in provide_data]
        label_names = [k[0] for k in provide_label]
        self._mod = mx.module.Module(symbol, data_names=data_names, label_names=label_names, context=ctx)
        self._mod.bind(provide_data, for_training=False)          
        self._mod.set_params(arg_params, aux_params)
        
    def predict(self, data_batch):
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))

class MutablePredictor(object):
    def __init__(self, symbol, prefix, epoch, provide_data, provide_label=[], ctx=mx.cpu(), arg_params=None, aux_params=None):
        data_names = [k[0] for k in provide_data]
        label_names = [k[0] for k in provide_label]
        self._mod = MutableModule(symbol, data_names, label_names, context=ctx)
        self._mod.bind(provide_data, for_training=False)
        if arg_params is None:
            arg_params, aux_params = load_param(prefix, epoch, convert=True, ctx=ctx, process=True)
        self._mod.set_params(arg_params, aux_params)
        
        self.symbol = symbol
        self.ctx = ctx
        
    def predict(self, data_batch):            
        self._mod.forward(data_batch)
        return dict(zip(self._mod.output_names, self._mod.get_outputs()))