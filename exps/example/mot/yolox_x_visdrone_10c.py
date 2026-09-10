# encoding: utf-8
#
# DARE-MOT 10-class detector exp (2026-09-10). Same as yolox_x_visdrone_mc.py except:
#   1. num_classes=10: the 5 evaluated classes keep model ids 0..4 (pedestrian, car, van, truck, bus);
#      people, bicycle, tricycle, awning-tricycle, motor are ids 5..9 and are dropped at tracking time.
#      The 5-class detector was trained with those five labelled BACKGROUND although they look like
#      pedestrians/cars -- 26% of CA-DARE's FPs sit on them (experiment-log 2026-09-10).
#   2. Ignored regions (raw cat 0/11) painted with the ImageNet mean colour during TRAINING, and GT
#      boxes >= 50% inside them dropped (tools/convert_visdrone_10c.py) -- official dropObjects rule.
#   3. Val stays the 5 evaluated classes (val7_10c.json = val7_mc.json + 10-entry category list), so
#      the AP printed during training is comparable with the 5-class detector's.
# Epochs via env DARE_10C_EPOCHS (default 3, ~5.4 h each on the RTX 5060 Ti at b=4).
#   python tools/train.py -f exps/example/mot/yolox_x_visdrone_10c.py -d 1 -b 4 --fp16 -c <init ckpt>
import os
import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp

# ImageNet mean, BGR 0..255 (cv2 images are BGR) -- what the network sees as "no signal"
IGNORE_FILL = (104, 116, 124)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.train_ann = "train_10c.json"   # 56-seq disjoint train split, 10 classes + ignore regions
        self.val_ann = "val7_10c.json"      # 7-seq val, 5 evaluated classes, 10-entry category list

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)

        self.max_epoch = int(os.environ.get("DARE_10C_EPOCHS", "3"))
        self.no_aug_epochs = 0              # avoids the Windows spawn deadlock (as in the MC exp)
        self.warmup_epochs = 1
        self.print_interval = 20
        self.eval_interval = 1

        self.basic_lr_per_img = 0.001 / 64.0

        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.data_num_workers = 2

        self.train_data_dir = r"C:\Users\User\Desktop\datasets\VisDrone2019-MOT-train"
        self.data_dir = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val"
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
            ignore_fill=IGNORE_FILL,
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
