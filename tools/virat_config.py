from core.config import cfg

# ----------------------------------------------------------------------
def set_virat_configs():
    cfg.MODEL.CLIP_GRADIENT = True
    
    # cfg.MODEL.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.CLS_AGNOSTIC_BBOX_REG = False #default

    # cfg.MODEL.NUM_CLASSES = 3 #background, sedan, suv
    cfg.MODEL.NUM_CLASSES = 3 #background, person, car

    cfg.COLOR.NUM_CLASSES = 6 #(yellow for person)
    cfg.COLOR.LOSS_WEIGHT = 0.5 # 1 is defualt

    cfg.ROTATION.NUM_CLASSES = int( (360-0)/10 )
    cfg.ROTATION.LOSS_WEIGHT = 2.0 # 1 is defualt

    cfg.X.NUM_CLASSES = int( (1 - (-1))/0.1 )
    cfg.X.LOSS_WEIGHT = 1.0 # 1 is defualt

    cfg.Y.NUM_CLASSES = int( (1 - (-1))/0.1 )
    cfg.Y.LOSS_WEIGHT = 1.0 # 1 is defualt

    cfg.DEPTH.WIDTH = 426 #has to be int
    cfg.DEPTH.HEIGHT = 240        
    cfg.DEPTH.NUM_CLASSES = 64
    cfg.DEPTH.LOSS_WEIGHT = 1.0 # 1 is defualt

    cfg.NORMAL.WIDTH = 426 #has to be int
    cfg.NORMAL.HEIGHT = 240        
    cfg.NORMAL.NUM_CLASSES = 10**3
    cfg.NORMAL.LOSS_WEIGHT = 1.0 # 1 is defualt

    cfg.TRAIN.FG_THRESH = 0.5  #default
    # cfg.TRAIN.FG_THRESH = 0.9

    return 

# ----------------------------------------------------------------------
