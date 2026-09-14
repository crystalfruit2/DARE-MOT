# encoding: utf-8
#
# DARE-MOT 5-class CONTROL for the clean 10-class detector (2026-09-14).
#
# The clean run (yolox_x_visdrone_10c_mot17init) changed the init AND the class set at once, and recall
# fell 14-17% while FP fell 30% (experiment-log 2026-09-11 cont. 4, read #4). This exp is the control
# that separates them: everything is held identical to yolox_x_visdrone_10c.py -- same MOT17 init
# (YOLOX_outputs/_init/yolox_x_10c_from_mot17.pth.tar, cls_preds dropped, so the 5-row head reinits the
# same way the 10-row one did), same 3 epochs, same warmup, same b=4/fp16, same input_size and the same
# random_size (18, 32) multiscale range, same ignore-region painting and dropObjects GT filtering --
# and ONLY the class set changes:
#   train_5c.json = train_10c.json with categories 6..10 (people, bicycle, tricycle, awning-tricycle,
#   motor) removed, built by _make_5c_train_json.py. Same 24,201 images, same 63,843 ignore regions,
#   817,593 boxes.
# NOTE on speed: the 09-11 note suggested capping multiscale at ~896 to avoid the sysmem spill. Not done
# here on purpose -- a different augmentation range would re-confound the comparison, and the 10-class
# run finished in 9 h 49 min at (18, 32) anyway (the "~16 GPU-h" estimate predates that measurement).
# Val is val7_mc.json: identical GT to val7_10c.json, so the printed AP50 is comparable with 56.0/60.7.
# Epochs via env DARE_5C_EPOCHS (default 3).
#   python tools/train.py -f exps/example/mot/yolox_x_visdrone_5c_ctrl.py -d 1 -b 4 --fp16 -c <init ckpt>
import os
import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp

# ImageNet mean, BGR 0..255 (cv2 images are BGR) -- what the network sees as "no signal"
IGNORE_FILL = (104, 116, 124)


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 5
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.train_ann = "train_5c.json"    # same images, same ignore regions, categories 6..10 dropped
        self.val_ann = "val7_mc.json"       # identical GT to val7_10c.json, 5-entry category list

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)

        self.max_epoch = int(os.environ.get("DARE_5C_EPOCHS", "3"))
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
