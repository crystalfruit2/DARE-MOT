import trackeval
import os

TE_ROOT = os.path.dirname(os.path.abspath(__file__))
SPLIT = "VisDrone339-val"

dataset_config = {
    'GT_FOLDER': os.path.join(TE_ROOT, 'gt', 'mot_challenge'),
    'TRACKERS_FOLDER': os.path.join(TE_ROOT, 'trackers', 'mot_challenge'),
    'BENCHMARK': 'VisDrone339',
    'SPLIT_TO_EVAL': 'val',
    'SEQMAP_FOLDER': os.path.join(TE_ROOT, 'gt', 'mot_challenge', 'seqmaps'),
    'TRACKERS_TO_EVAL': ['bytetrack', 'static_ema', 'dare_A_lockoff', 'dare_B_lockon'],
    'CLASSES_TO_EVAL': ['pedestrian'],
    'SKIP_SPLIT_FOL': False,
    'PRINT_CONFIG': False,
}

eval_config = trackeval.Evaluator.get_default_eval_config()
eval_config['PRINT_RESULTS'] = True
eval_config['PRINT_CONFIG'] = False
eval_config['DISPLAY_LESS_PROGRESS'] = True

evaluator = trackeval.Evaluator(eval_config)
dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
metrics = [trackeval.metrics.HOTA()]

results, msg = evaluator.evaluate([dataset], metrics)

for tracker in dataset_config['TRACKERS_TO_EVAL']:
    hota_res = results['MotChallenge2DBox'][tracker]['uav0000339_00001_v']['pedestrian']['HOTA']
    print(f"\n=== {tracker} ===")
    for k in ['HOTA', 'DetA', 'AssA']:
        print(f"{k}: {hota_res[k].mean():.4f}")
