import time
import logging
import mxnet as mx

class Speedometer(object):
    """Calculate and long training speed prriodically ON ONE LINE.
    
    Parameters
    ----------
    batch_size: int
        batch_size of data
    frequent: int
        how many batches between calculations.
    """
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    lr = param.locals['optimizer_params']['lr_scheduler'].base_lr
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec, lr:%f\tTrain-" % (param.epoch, count, speed, lr)
                    for n, v in zip(name, value):
                        s += "%s=%.5f,\t" % (n, v)
                    logging.info(s)
                else:
                    logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


def do_checkpoint(prefix, means, stds):
    def _callback(iter_no, sym, arg, aux):
        arg['bbox_pred_weight_test'] = (arg['bbox_pred_weight'].T * mx.nd.array(stds)).T
        arg['bbox_pred_bias_test'] = arg['bbox_pred_bias'] * mx.nd.array(stds) + mx.nd.array(means)
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
        arg.pop('bbox_pred_weight_test')
        arg.pop('bbox_pred_bias_test')
    return _callback


def do_checkpoint_rpn(prefix):
    def _callback(iter_no, sym, arg, aux):
        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
    return _callback