# encoding: utf-8
#
# DARE-MOT multi-class TRACKING exp for the val7 benchmark (Phase 3/4 re-measure).
#
# Same detector geometry + 5-class head as the training exp (yolox_x_visdrone_mc.py), but
# wired for tools/track.py: only the eval side matters. tracks the 7 held-out val sequences
# with the multi-class YOLOX-X detector so the tracker can emit per-detection class, which
# _score_multiclass.py then splits into per-class MOTChallenge metrics.
#
# Run (after the detector finishes training):
#   python tools/track.py -f exps/example/mot/yolox_x_visdrone_mc_val7.py \
#       -c YOLOX_outputs/yolox_x_visdrone_mc/best_ckpt.pth.tar -b 1 -d 1 --fp16 --fuse \
#       --seed 0 --track_thresh 0.6 --track_buffer 30 --match_thresh 0.9 --min-box-area 100 \
#       -expn <run-name>
import os
import torch

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 5           # pedestrian, car, van, truck, bus (MC ids 1..5 = head+1)
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.val_ann = "val7_mc.json"      # 7-seq val benchmark set (5 classes)

        self.input_size = (800, 1440)
        self.test_size = (800, 1440)

        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.data_num_workers = 2

        # Raw VisDrone val root: annotations/val7_mc.json + sequences/<seq>/<img>.
        self.data_dir = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone2019-MOT-val"
        self.img_name = "sequences"

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
