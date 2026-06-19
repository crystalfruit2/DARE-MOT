# Confirm the detector works by running the SAME COCOEvaluator on matched-domain
# (pedestrian) data instead of the mismatched car sequence.
# NOTE: this evals on data the model trained on, so AP is optimistic — but it
# definitively separates "detector works" from "detector broken".
import torch
from yolox.exp import get_exp

EXP = r"exps/example/mot/yolox_x_visdrone_finetune.py"
CKPT = r"YOLOX_outputs/yolox_x_visdrone_finetune/latest_ckpt.pth.tar"
TRAIN_DIR = r"C:\Users\User\Desktop\projects\ByteTrack\datasets\VisDrone_MOT_Format\VisDrone2019-MOT-val"

def main():
    exp = get_exp(EXP, None)
    exp.data_num_workers = 0
    exp.data_dir = TRAIN_DIR          # eval on the pedestrian data
    exp.val_ann = "train.json"
    model = exp.get_model().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    evaluator = exp.get_evaluator(batch_size=1, is_distributed=False)
    ap50_95, ap50, summary = evaluator.evaluate(model, False, True)
    print("\n========== MATCHED-DOMAIN (pedestrian) EVAL ==========")
    print(f"AP@[.5:.95] = {ap50_95:.4f}")
    print(f"AP@.50      = {ap50:.4f}")
    print(summary)

if __name__ == "__main__":
    main()
