"""
given a pascal voc imdb, compute mAP
"""

import numpy as np
import os
import cPickle


def parse_voc_rec(filename):
    """
    parse pascal voc record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['name'] = obj.find('name').text
        obj_dict['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(float(bbox.find('xmin').text)),
                            int(float(bbox.find('ymin').text)),
                            int(float(bbox.find('xmax').text)),
                            int(float(bbox.find('ymax').text))]
        objects.append(obj_dict)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imageset_file, classname, annocache, ovthresh=0.5, use_07_metric=False):
    """
    pascal voc evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param classname: category name
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :param use_07_metric: whether to use voc07's 11 point ap computation
    :return: rec, prec, ap
    """
    area_names = ['all', '0-25', '25-50', '50-100',
                      '100-200', '200-300', '300-inf']
    area_ranges = [[0**2, 1e5**2], [0**2, 25**2], [25**2, 50**2], [50**2, 100**2],
                   [100**2, 200**2], [200**2, 300**2], [300**2, 1e5**2]]

    with open(imageset_file, 'r') as f:
        lines = f.readlines()
    image_filenames = [x.strip() for x in lines]

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = {}
        for ind, image_filename in enumerate(image_filenames):
            recs[image_filename] = parse_voc_rec(annopath.format(image_filename))
            if ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(image_filenames))
        print 'saving annotations cache to {:s}'.format(annocache)
        with open(annocache, 'wb') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'rb') as f:
            recs = cPickle.load(f)

    # extract objects in :param classname:
    class_recs = {}
    npos = 0
    num_sample = np.zeros((7,), dtype=int)
    for image_filename in image_filenames:
        objects = [obj for obj in recs[image_filename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in objects])
        difficult = np.array([x['difficult'] for x in objects]).astype(np.bool)
        det = [False] * len(objects)  # stand for detected
        npos = npos + sum(~difficult)
        class_recs[image_filename] = {'bbox': bbox,
                                      'difficult': difficult,
                                      'det': det}
        f_bbox = bbox.astype(float)
        if f_bbox.size>0:
            area = (f_bbox[:,2]-f_bbox[:,0])*(f_bbox[:,3]-f_bbox[:,1])
            for i in range(7):
                for j in range(area.shape[0]):
                    num_sample[i] += (area[j]>area_ranges[i][0] and area[j]<area_ranges[i][1])
                 #min(np.sum(list()), np.sum(list()))
    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    bbox = np.array([[float(z) for z in x[2:]] for x in splitlines])

    area = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])
    index_max = np.where(area > area_ranges[1][1])
    index_min = np.where(area < area_ranges[1][1])

    image_ids_ = [x[0] for x in splitlines]
    confidence_ = np.array([float(x[1]) for x in splitlines])

    confidence_min = np.array([confidence_[i] for i in index_min[0]])
    sorted_inds = np.argsort(-confidence_min)
    index_min = index_min[0][:int(0.0*float(sorted_inds.shape[0]))]
    index_max = np.concatenate((index_max[0],index_min),axis=0)

    image_ids = [image_ids_[i] for i in index_max]
    confidence = np.array([confidence_[i] for i in index_max])
    bbox = np.array(bbox[index_max,:])


    # sort by confidence
    if bbox.shape[0] > 0:
        sorted_inds = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        bbox = bbox[sorted_inds, :]
        image_ids = [image_ids[x] for x in sorted_inds]

    # go down detections and mark true positives and false positives
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_index=np.zeros((7,))
    fp_index = np.zeros((7,))
    size_index=np.ones(nd)*-1
    for d in range(nd):
        r = class_recs[image_ids[d]]
        bb = bbox[d, :].astype(float)
        area = (bb[2]-bb[0])*(bb[3]-bb[1])
        #if abs(sorted_scores[d]) > 0.5:
        # if area < area_ranges[0][1] and area > area_ranges[0][0]:
        #     size_index[d]=0
        if area < area_ranges[1][1]:
            size_index[d]=1
        elif area < area_ranges[2][1]:
            size_index[d]=2
        elif area < area_ranges[3][1]:
            size_index[d]=3
        elif area < area_ranges[4][1]:
            size_index[d]=4
        elif area < area_ranges[5][1]:
            size_index[d]=5
        elif area < area_ranges[6][1]:
            size_index[d]=6
        ovmax = -np.inf
        bbgt = r['bbox'].astype(float)

        if bbgt.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not r['difficult'][jmax]:
                box = (r['bbox'][jmax]).astype(float)#get gt box
                area = (box[2] - box[0]) * (box[3] - box[1])
                if not r['det'][jmax]:
                    tp[d] = 1.
                    r['det'][jmax] = 1
                    tp_index[0]+=1
                    if area < area_ranges[1][1]:
                        tp_index[1] += 1
                    elif area < area_ranges[2][1]:
                        tp_index[2] += 1
                    elif area < area_ranges[3][1]:
                        tp_index[3] += 1
                    elif area < area_ranges[4][1]:
                        tp_index[4] += 1
                    elif area < area_ranges[5][1]:
                        tp_index[5] += 1
                    elif area < area_ranges[6][1]:
                        tp_index[6] += 1
                else:
                    fp[d] = 1.
                    fp_index[0] += 1
                    if area < area_ranges[1][1]:
                        fp_index[1] += 1
                    elif area < area_ranges[2][1]:
                        fp_index[2] += 1
                    elif area < area_ranges[3][1]:
                        fp_index[3] += 1
                    elif area < area_ranges[4][1]:
                        fp_index[4] += 1
                    elif area < area_ranges[5][1]:
                        fp_index[5] += 1
                    elif area < area_ranges[6][1]:
                        fp_index[6] += 1

        else:
            fp[d] = 1.

    # compute precision recall
    index = list()
    #num_sample = list()
    num_fp = list()
    num_tp = list()
    #print size_index
    #print 'rec of {:s}:{:.3f}'.format(area_names[0], float(tp_index[0])/max(0.001, float(num_sample[0])))
    #print 'prec of {:s}:{:.3f}'.format(area_names[0], float(tp_index[0])/max(0.001, float(tp_index[0]+fp_index[1])))
    for i in range(7):
        print 'gt rec of {:s}:{:.3f}'.format(area_names[i], float(tp_index[i])/max(0.001, float(num_sample[i])))
        print 'gt prec of {:s}:{:.3f}'.format(area_names[i], float(tp_index[i])/max(0.001, float(tp_index[i]+fp_index[i])))
        print 'percent of {:s}:{:.3f}'.format(area_names[i], float(np.sum(size_index==i))/(float(nd)+0.001))
        print 'detect prec of {:s}:{:.3f}'.format(area_names[i], float(np.sum(tp[size_index==i]))/(float(np.sum(tp[size_index==i])+np.sum(fp[size_index==i]))+0.001))
        fp_ = np.cumsum(fp[size_index==i])
        tp_ = np.cumsum(tp[size_index==i])
        rec_ = tp_ / float(num_sample[i])
        prec_ = tp_ / np.maximum(tp_ + fp_, np.finfo(np.float64).eps)
        ap_ = voc_ap(rec_, prec_, use_07_metric)
        print('AP for {} = {:.4f}'.format(area_names[i], ap_))


    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid division by zero in case first detection matches a difficult ground ruth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)




    print '######################################################'
    return rec, prec, ap
