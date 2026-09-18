# encoding: utf-8
#
# DARE-MOT TRACKING exp for the 10-class detector (yolox_x_visdrone_10c*, 2026-09-11) on val7.
# Identical to yolox_x_visdrone_mc_val7.py except the head has 10 classes and the val json carries
# the 10-entry category list. Model ids 0..4 = the 5 evaluated classes, so run the tracker with
# DARE_MAX_CLASS=4 to drop people/bicycle/tricycle/awning-tricycle/motor before association.
#   python tools/track.py -f exps/example/mot/yolox_x_visdrone_10c_val7.py \
#       -c YOLOX_outputs/yolox_x_visdrone_10c_mot17init/best_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse \
#       --seed 0 --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 -expn <run>
import os

from yolox_x_visdrone_mc_val7 import Exp as MCVal7Exp


class Exp(MCVal7Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.val_ann = "val7_10c.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
