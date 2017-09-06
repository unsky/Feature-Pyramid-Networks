import time
import logging
import mxnet as mx

from rcnn_predict import RCNNPredictor

def do_eval(network, output_folder, ctx, config, imdb, roidb, fpn=False, step='rcnn'):
    def _callback(iter_no, sym, arg, aux):
        predictor = RCNNPredictor(network, output_folder, iter_no + 1, ctx, config, fpn=fpn, step=step)
        predictor.imdb_eval(imdb, roidb)
    return _callback 
