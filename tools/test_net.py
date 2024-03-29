"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import utils.logging
from virat_config import set_virat_configs

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--test_dataset',
        help='testing dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        ckpt_num = ckpt_path.replace('.pth','').replace('model_step', '').split('/')[-1]
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test', ckpt_num, args.test_dataset)
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis
    cfg.VIS_TH = 0.5

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    cfg.TEST.FORCE_JSON_DATASET_EVAL = True
    if args.dataset == "coco2017":         
        cfg.MODEL.NUM_CLASSES = 81
        args.dataset = args.test_dataset
        cfg.TEST.DATASETS = (args.test_dataset,)
    # ----------------------------------------------
    elif args.dataset == "virat1":
        set_virat_configs()
        cfg.TEST.DATASETS = ('virat1_real_val',)
    # ----------------------------------------------
    elif args.dataset == "virat2":
        set_virat_configs()
        cfg.TEST.DATASETS = ('virat2_real_val',)
    # ----------------------------------------------
    elif args.dataset == "epfl":
        set_virat_configs()
        cfg.TEST.DATASETS = ('epfl_real_val',)
    # ----------------------------------------------
    elif args.dataset == "uadetrac1":
        set_virat_configs()
        cfg.TEST.DATASETS = ('uadetrac1_real_val',)
    # ----------------------------------------------
    elif args.dataset == "uadetrac2":
        set_virat_configs()
        cfg.TEST.DATASETS = ('uadetrac2_real_val',)
    # ----------------------------------------------
    elif args.dataset == "car_coco":
        set_virat_configs()
        cfg.TEST.DATASETS = (args.test_dataset,)
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)
