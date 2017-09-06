import time
import logging
import mxnet as mx

from rpn_predict import RPNPredictor

def do_eval(network, output_folder, ctx, config, imdb, roidb, fpn=False):
    def _callback(iter_no, sym, arg, aux):
        predictor = RPNPredictor(network, output_folder, iter_no + 1, ctx, config, fpn=fpn)
        predictor.imdb_eval(imdb, roidb)
    return _callback 
