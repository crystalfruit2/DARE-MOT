# encoding: utf-8
#
# DARE-MOT multiclass detector exp (Phase 2 of the multiclass migration).
#
# Fixes BOTH leaks the single-class yolox_x_visdrone_finetune.py had:
#   1. class-leak    : num_classes=5, real categories (was 1, hardcoded pedestrian).
#   2. det train/test: trains on the OFFICIAL DISJOINT 56-seq VisDrone train split
#                      (was train6.json = 6 of the 7 VAL sequences it's benchmarked on).
#
# Data comes from the Phase-1 MC pipeline (convert_visdrone_mc.py + _mot_to_coco_mc.py):
#   train_mc.json : 56 train seqs, 826,697 boxes, 5 classes  (disjoint from val)
#   val7_mc.json  : 7  val   seqs, 72,690  boxes, 5 classes  (the tracker benchmark set)
# file_name is "<seq>/<img>", so data_dir=<raw root>, name="sequences".
#
# Warm-start (recommended): pass --ckpt to the 1-class best_ckpt; load_ckpt() skips the
# shape-mismatched head and keeps the backbone. Head reinits to 5 classes.
#   python tools/train.py -f exps/example/mot/yolox_x_visdrone_mc.py -d 1 -b 4 --fp16 \
#       -c YOLOX_outputs/yolox_x_visdrone_finetune/best_ckpt.pth.tar
import os
import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 5           # pedestrian, car, van, truck, bus
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.train_ann = "train_mc.json"   # 56-seq disjoint train split
        self.val_ann = "val7_mc.json"      # 7-seq val (tracker benchmark set)

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)

        # 56 seqs / 24,201 imgs — ~10x the old (val-subset) train set. Start modest;
        # raise after watching AP. no_aug_epochs=0 avoids the Windows spawn deadlock.
        self.max_epoch = 8
        self.no_aug_epochs = 0
        self.warmup_epochs = 1
        self.print_interval = 20
        self.eval_interval = 1

        self.basic_lr_per_img = 0.001 / 64.0

        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.data_num_workers = 2

        # Images referenced in place (no duplication). file_name = "<seq>/<img>".
        # TRAIN: official disjoint 56-seq split.
        self.train_data_dir = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train"
        # VAL: 7 held-out val seqs (disjoint from train).
        self.data_dir = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val"

        # MOTDataset joins images as data_dir/name/file_name -> <root>/sequences/<seq>/<img>
        self.img_name = "sequences"

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=self.train_data_dir,
            json_file=self.train_ann,
            name=self.img_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name=self.img_name,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
