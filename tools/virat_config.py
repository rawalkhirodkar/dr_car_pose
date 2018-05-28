from core.config import cfg

# ----------------------------------------------------------------------
def set_virat_configs():
    cfg.MODEL.CLIP_GRADIENT = True

    cfg.MODEL.ATTRIBUTE_ON = True
    # cfg.MODEL.ATTRIBUTE_ON = False

    # cfg.MODEL.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.CLS_AGNOSTIC_BBOX_REG = False

    cfg.MODEL.NUM_CLASSES = 3 #background, sedan, suv
    cfg.MODEL.COLOR_NUM_CLASSES = 6 #(yellow for person)
    cfg.MODEL.ROTATION_NUM_CLASSES = int( (360-0)/10 )
    cfg.MODEL.X_NUM_CLASSES = int( (1 - (-1))/0.1 )
    cfg.MODEL.Y_NUM_CLASSES = int( (1 - (-1))/0.1 )

    cfg.MODEL.DEPTH_WIDTH = 320 #has to be int
    cfg.MODEL.DEPTH_HEIGHT = 192        
    cfg.MODEL.DEPTH_NUM_CLASSES = 64

    cfg.MODEL.NORMAL_WIDTH = 320 #has to be int
    cfg.MODEL.NORMAL_HEIGHT = 192        
    cfg.MODEL.NORMAL_NUM_CLASSES = 10**3

    cfg.TRAIN.FG_THRESH = 0.5  #default
    # cfg.TRAIN.FG_THRESH = 0.9

    return 

# ----------------------------------------------------------------------
