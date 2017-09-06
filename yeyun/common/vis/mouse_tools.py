import cv2
import numpy as np

from vis_im import draw_bbox, draw_points, show_text

got_bbox = False
drawing_bbox = False
bbox = [0, 0, 0, 0]

def mouse_handle(event, x, y, flags, param):
    global drawing_bbox
    global bbox
    global got_bbox
    if event == cv2.EVENT_MOUSEMOVE:
        if drawing_bbox:
            bbox[2] = x
            bbox[3] = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing_bbox = True
        bbox = [x, y, x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        if bbox[0] > bbox[2]:
            bbox[0], bbox[2] = bbox[2], bbox[0]
        if bbox[1] > bbox[3]:
            bbox[1], bbox[3] = bbox[3], bbox[1]
        drawing_bbox = False
        got_bbox = True

org = np.array([])
img = np.array([])
tmp = np.array([])
window_name = 'test'
region = [0, 0, 0, 0]
pre_pt = (-1, -1)
def on_mouse(event, x, y, flags, param):
    """ Mouse callback for video """
    global org
    global img
    global tmp
    global window_name
    global region
    global pre_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        img = tmp.copy()
        pre_pt = (x, y)
        
        region[0] = x
        region[1] = y
        
        show_text(img, "({},{})".format(x, y), pre_pt)
        draw_points(img, pre_pt)
        cv2.imshow(window_name, img)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        tmp = img.copy()
        cur_pt = (x, y)
        show_text(tmp, "({},{})".format(x, y), cur_pt)
        bbox = [pre_pt[0], pre_pt[1], cur_pt[0], cur_pt[1]]
        draw_bbox(tmp, bbox)
        cv2.imshow(window_name, tmp)
    elif event == cv2.EVENT_MOUSEMOVE and not (flags & cv2.EVENT_FLAG_LBUTTON):
        tmp = img.copy()
        cur_pt = (x, y)
        show_text(tmp, "({},{})".format(x, y), cur_pt)
        cv2.imshow(window_name, tmp)
    elif event == cv2.EVENT_LBUTTONUP:
        img = org.copy()
        x = min(x, img.shape[1] - 1)
        y = min(y, img.shape[0] - 1)
        cur_pt = (x, y)
        region[2] = x
        region[3] = y
        
        show_text(img, "({},{})".format(x, y), cur_pt)
        draw_points(img, pre_pt)
        bbox = [pre_pt[0], pre_pt[1], cur_pt[0], cur_pt[1]]
        draw_bbox(img, bbox)
        cv2.imshow(window_name, img)
        tmp = img.copy()
                    
def draw_boxes_on_image(image, pause, c):
    """ draw boxes on image """
    global org
    global img
    global tmp
    global window_name
    global region
    
    org = image.copy()
    img = org.copy()
    tmp = img.copy()
    
    boxes = list()
    
    if pause == 0:
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, on_mouse, 0)
        cv2.imshow(window_name, image)
        
    c = cv2.waitKey(pause)
    if c == ord('p'):
        pause = not pause
    #     cv2.setMouseCallback(window_name, 0, 0)
    if region[0] < region[2]:
        draw_bbox(image, region)
        boxes.append(region)
    
    region = [0, 0, 0, 0]
    return boxes, pause, c