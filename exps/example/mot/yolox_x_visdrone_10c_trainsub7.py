# encoding: utf-8
#
# DARE-MOT TRACKING exp for the out-of-val tracker-ranking check (2026-09-15): the clean 10-class detector D3
# on 7 VisDrone-MOT TRAIN sequences (every 8th by name; _make_trainsub_2026-09-15.py). D3 was trained on these
# frames, so only tracker rankings are meaningful here. Identical to yolox_x_visdrone_10c_val7.py otherwise;
# run with DARE_MAX_CLASS=4.
import os

from yolox_x_visdrone_10c_val7 import Exp as TenClassVal7Exp


class Exp(TenClassVal7Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train"
        self.val_ann = "trainsub7_10c.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
