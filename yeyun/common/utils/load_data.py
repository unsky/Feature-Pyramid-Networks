import cv2
from multiprocessing import Pool

from ..dataset import *



def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb

def read_image(image_name):
    return cv2.imread(image_name)

def load_imageset(dataset, image_set, dataset_root_path, cache_path,
                  filter_strategy, category='all', flip=False, cache_image=False, task='detection'):
    # load dataset and prepare imdb for training
    
    image_sets = [iset for iset in image_set.split('+')]
    print (image_sets)

    if dataset == 'coco':
        imdbs = [coco(image_set, cache_path, dataset_root_path, category, task)
                 for image_set in image_sets]
        roidbs = [imdb.gt_roidb(debug=False) for imdb in imdbs]
    if dataset == 'PascalVOC':
        imdbs = [PascalVOC(image_set, cache_path, dataset_root_path, category=category) for image_set in image_sets]
        roidbs = [imdb.gt_roidb() for imdb in imdbs]
    if dataset == 'kitti':
        imdbs = [kitti(image_set, cache_path, dataset_root_path, category=category) for image_set in image_sets]
        roidbs = [imdb.gt_roidb() for imdb in imdbs]
                
    if flip:
        roidbs = [imdb.append_flipped_images(roidb) for imdb, roidb in zip(imdbs, roidbs)]
    
    if filter_strategy.remove_empty:
        roidbs = [filter_empty_roidb(roidb) for roidb in roidbs]
    if filter_strategy.remove_multi:
        roidbs = [filter_multi_roidb(roidb) for roidb in roidbs]
    if filter_strategy.remove_unvis:  
        roidbs = [filter_unvis_roidb(roidb) for roidb in roidbs]    
        
    roidbs = [filter_point_roidb(roidb) for roidb in roidbs]

    roidb = merge_roidb(roidbs) 
 
    if cache_image:        
        image_list = [roi_vec['image'] for roi_vec in roidb]
        pool = Pool()
        print len(image_list)
        for i, res in enumerate(pool.imap(read_image, image_list)):
            if i % 300 == 0:
                print '{}/{}'.format(i, len(image_list))
            roidb[i]['cache_image'] = res
        
#         for i in range(len(roidb)):
#             if i % 100 == 0:
#                 print '{}/{}'.format(i, len(roidb))
#             roidb[i]['cache_image'] = cv2.imread(roidb[i]['image'])
    
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb

def sample_roidb(roidb, sample):
    roidb = roidb[::sample]

def filter_point_roidb(roidb):
    """ remove roidb with entries with point as gt box """
    def not_point_as_bbox(entry):
        gt_boxes = entry['boxes']
        width = gt_boxes[:, 2] - gt_boxes[:, 0]
        height = gt_boxes[:, 3] - gt_boxes[:, 1]
        return (width > 0).all() and (height > 0).all
    
    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if not_point_as_bbox(entry)]
    num_after = len(filtered_roidb)
    print 'filtered %d point roidb entries: %d -> %d' % (num - num_after, num, num_after)
    return filtered_roidb
     

def filter_empty_roidb(roidb):
    """ remove roidb entries without gt box """
    
    def not_empty(entry):
        """ valid images have at least 1 gt """
        gt_boxes = entry['boxes']
        return gt_boxes.shape[0] > 0
    
    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if not_empty(entry)]
    num_after = len(filtered_roidb)
    print 'filtered %d empty roidb entries: %d -> %d' % (num - num_after, num, num_after)
    return filtered_roidb
        
def filter_multi_roidb(roidb):
    """ remove roidb entries with multi gt boxes """
    def single_roidb(entry):
        """ valid images have only one gt """
        gt_boxes = entry['boxes']
        return gt_boxes.shape[0] <= 1
    
    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if single_roidb(entry)]
    num_after = len(filtered_roidb)
    print 'filtered %d multi roidb entries: %d -> %d' % (num - num_after, num, num_after)
    return filtered_roidb

def filter_unvis_roidb(roidb):
    """ remove roidb entries with un-visible points, special for coco keypoint """
    def all_point_vis(entry):
        point = entry['keypoints'][:, 2::3]
        return (point==2).all()
    
    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if all_point_vis(entry)]
    num_after = len(filtered_roidb)
    print 'filter %d un-visible roidb entries: %d -> %d' % (num - num_after, num, num_after)
    return filtered_roidb

