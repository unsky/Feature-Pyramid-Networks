import mxnet as mx
import numpy as np


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names():
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if True:
        pred.append('rcnn_label')
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label

def get_fpn_rpn_names():
    pred = []
    label = []
    rpn_depth = 4
    for i in range(rpn_depth):
        pred.append('rpn_cls_prob_p%d'%(i+2))
    for i in range(rpn_depth):
        pred.append('rpn_bbox_loss_p%d'%(i+2))
    for i in range(rpn_depth):
        label.append('rpn_label_p%d'%(i+2))
    for i in range(rpn_depth):
        label.append('rpn_bbox_target_p%d'%(i+2))
    for i in range(rpn_depth):
        label.append('rpn_bbox_weight_p%d'%(i+2))
    return pred, label

def get_fpn_rcnn_names():
    pred = []
    label = []
    rcnn_depth = 4
    for i in range(rcnn_depth):
        pred.append('rcnn_cls_prob_p%d'%(i+2))
    for i in range(rcnn_depth):
        pred.append('rcnn_bbox_loss_p%d'%(i+2))
    if True:
        for i in range(rcnn_depth):
            pred.append('rcnn_label_p%d'%(i+2))
        rpn_pred, rpn_label = get_fpn_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label

class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class FPNRPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRPNAccMetric, self).__init__('FPNRPNAcc')
        self.pred, self.label = get_fpn_rpn_names()
        self.rpn_depth = 4
    def update(self, labels, preds):
        for i in range(self.rpn_depth):
            #print self.label
            pred = preds[self.pred.index('rpn_cls_prob_p%d'%(i+2))]

            label = labels[self.label.index('rpn_label_p%d'%(i+2))]

            # pred (b, c, p) or (b, c, h, w)
            pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
            pred_label = pred_label.reshape((pred_label.shape[0], -1))
            # label (b, p)
            label = label.asnumpy().astype('int32')

            # filter with keep_inds
            keep_inds = np.where(label != -1)
            pred_label = pred_label[keep_inds]
            label = label[keep_inds]

            self.sum_metric += np.sum(pred_label.flat == label.flat)
            self.num_inst += len(pred_label.flat)

class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = True
        self.pred, self.label = get_rcnn_names()


    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')
        print "pred",pred_label
        print 'label',label
        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class FPNRCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRCNNAccMetric, self).__init__('FPNRCNNAcc')
        self.e2e = True
        self.pred, self.label = get_fpn_rcnn_names()
        self.rcnn_depth = 4
    def update(self, labels, preds):
        for i in range(self.rcnn_depth):
            pred = preds[self.pred.index('rcnn_cls_prob_p%d'%(i+2))]
            if self.e2e:
                label = preds[self.pred.index('rcnn_label_p%d'%(i+2))]
            else:
                label = labels[self.label.index('rcnn_label_p%d'%(i+2))]

            last_dim = pred.shape[-1]

            pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
            label = label.asnumpy().reshape(-1,).astype('int32')

            pred_flat = pred_label.flat
            label_flat = label.flat
            keep_inds = np.where(label_flat != -1)
            pred_flat = pred_flat[keep_inds]
            label_flat = label_flat[keep_inds]
            # print 'layer index: ', i
            # print 'predict      labels:'
            # for i in range(len(pred_flat)):
            #     print pred_flat[i], '       ', label_flat[i]
            self.sum_metric += np.sum(pred_flat == label_flat)
            self.num_inst += len(pred_flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class FPNRPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRPNLogLossMetric, self).__init__('FPNRPNLogLoss')
        self.pred, self.label = get_fpn_rpn_names()
        self.rpn_depth = 4
    def update(self, labels, preds):
        for i in range(self.rpn_depth):
            pred = preds[self.pred.index('rpn_cls_prob_p%d'%(i+2))]
            label = labels[self.label.index('rpn_label_p%d'%(i+2))]

            # label (b, p)
            label = label.asnumpy().astype('int32').reshape((-1))
            # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
            pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
            pred = pred.reshape((label.shape[0], -1))

            # filter with keep_inds
            keep_inds = np.where(label != -1)[0]
            label = label[keep_inds]
            cls = pred[keep_inds, label]

            cls += 1e-14
            cls_loss = -1 * np.log(cls)
            cls_loss = np.sum(cls_loss)
            self.sum_metric += cls_loss
            self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = True
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class FPNRCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRCNNLogLossMetric, self).__init__('FPNRCNNLogLoss')
        self.e2e = True
        self.pred, self.label = get_fpn_rcnn_names()
        self.rcnn_depth = 4
    def update(self, labels, preds):
        for i in range(self.rcnn_depth):
            pred = preds[self.pred.index('rcnn_cls_prob_p%d'%(i+2))]
            if self.e2e:
                label = preds[self.pred.index('rcnn_label_p%d'%(i+2))]
            else:
                label = labels[self.label.index('rcnn_label_p%d'%(i+2))]

            last_dim = pred.shape[-1]
            pred = pred.asnumpy().reshape(-1, last_dim)
            label = label.asnumpy().reshape(-1,).astype('int32')

            keep_inds = np.where(label != -1)[0]
            label = label[keep_inds]
            cls = pred[keep_inds, label]
            cls += 1e-14
            cls_loss = -1 * np.log(cls)
            cls_loss = np.sum(cls_loss)
            self.sum_metric += cls_loss
            self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()
        bbox_weight = labels[self.label.index('rpn_bbox_weight')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class FPNRPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRPNL1LossMetric, self).__init__('FPNRPNL1Loss')
        self.pred, self.label = get_fpn_rpn_names()
        self.rpn_depth = 4
    def update(self, labels, preds):
        for i in range(self.rpn_depth):
            bbox_loss = preds[self.pred.index('rpn_bbox_loss_p%d'%(i+2))].asnumpy()
            bbox_weight = labels[self.label.index('rpn_bbox_weight_p%d'%(i+2))].asnumpy()
            num_inst = np.sum(bbox_weight > 0) / 4

            self.sum_metric += np.sum(bbox_loss)
            self.num_inst += num_inst

class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = True
        self.pred, self.label = get_rcnn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.e2e:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class FPNRCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(FPNRCNNL1LossMetric, self).__init__('FPNRCNNL1Loss')
        self.e2e = True
        self.pred, self.label = get_fpn_rcnn_names()
        self.rcnn_depth = 4
    def update(self, labels, preds):
        for i in range(self.rcnn_depth):
            bbox_loss = preds[self.pred.index('rcnn_bbox_loss_p%d'%(i+2))].asnumpy()
            if self.e2e:
                label = preds[self.pred.index('rcnn_label_p%d'%(i+2))].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label_p%d'%(i+2))].asnumpy()
            num_inst = len(label[0])-len(np.where(label == 0)[0]) - len(np.where(label == -1)[0])
            self.sum_metric += np.sum(bbox_loss)
            self.num_inst += num_inst
