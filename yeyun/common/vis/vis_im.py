import cv2

def show_text(im, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5,
            color=(0, 0, 255), thick=1):
    """show text on image"""
    cv2.putText(im, text, (int(pos[0]), int(pos[1])), font, font_scale, color, thick)

def draw_bbox(im, boxes, color=(255, 0, 0), line=1, cls=None):
    """draw boxes on image
    
    Parameters:
    -----------
    boxes: numpy or list
        shape of numpy.[num, 4], numpy.[4]
        or list([4]), list([[4], [4], ...])
    """
    def draw_one_bbox(im, bbox, color, line):
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, line)
        if len(bbox) > 4:
            if cls is not None:
                show_text(im, '{}: {:.3f}'.format(cls, bbox[4]), (bbox[0], bbox[1]+10))
            else:
                show_text(im, 'socre: {:.3f}'.format(bbox[4]), (bbox[0], bbox[1]+10))
    
    try:
        [draw_one_bbox(im, bbox, color, line) for bbox in boxes]
    except:
        draw_one_bbox(im, boxes, color, line) 

def draw_bbox_center(im, boxes, radius=2, color=(255, 0, 0), line=2):
    """draw bbox center point
    
    """
    def draw_one_bbox_point(im, box, radius, color, line):
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        cv2.circle(im, (cx, cy), radius, color, line)
    [draw_one_bbox_point(im, box, radius, color, line) for box in boxes]
        
def draw_kps_points(im, points, radius=2, color=(0, 0, 255), line=2):
    """draw point on image
    
    Parameters:
    -----------
    points: numpy or list
    """
    def draw_one_instance(im, points, radius, color, line):
        for i in range(0, len(points), 3):
            x = points[i]
            y = points[i+1]
            cv2.circle(im, (x, y), radius, color, line)
    try:
        [draw_one_instance(im, point, radius, color, line) for point in points]
    except:
        draw_one_instance(im, points, radius, color, line)

def draw_points(im, points, radius=2, color=(0, 0, 255), line=2):
    """draw point on image
    
    Parameters:
    ----------
    points: numpy or list
        one point
    """
    def draw_one_point(im, point, radius, color, line):
        x = int(point[0])
        y = int(point[1])
        cv2.circle(im, (x, y), radius, color, line)
    try:
        [draw_one_point(im, point, radius, color, line) for point in points]
    except:
        draw_one_point(im, points, radius, color, line)

