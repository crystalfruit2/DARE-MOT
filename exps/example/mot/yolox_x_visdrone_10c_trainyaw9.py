# encoding: utf-8
#
# DARE-MOT TRACKING exp for the yaw-generalization check (2026-09-15): D3 on the 9 VisDrone-MOT TRAIN sequences
# that contain camera-yaw segments (_make_trainyaw_2026-09-15.py). D3 was trained on these frames: rankings and
# switch locations only. Run with DARE_MAX_CLASS=4.
import os

from yolox_x_visdrone_10c_val7 import Exp as TenClassVal7Exp


class Exp(TenClassVal7Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.data_dir = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train"
        self.val_ann = "trainyaw9_10c.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
