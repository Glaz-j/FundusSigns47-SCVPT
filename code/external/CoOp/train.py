import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.retinal_vision2

import trainers.coop
import trainers.cocoop
import trainers.coop_multilabel
import trainers.coop_multilabel_cached
import trainers.biomedcoop_multilabel_cached
import trainers.openclip_coop_multilabel_cached
import trainers.openclip_vpt_multilabel
import trainers.openclip_vpt_only_multilabel
import trainers.openclip_vpt_deep_only_multilabel
import trainers.openclip_maple_multilabel
import trainers.openclip_deepvpt_textconcat_multilabel
import trainers.openclip_deepvpt_classregion_multilabel
import trainers.openclip_lora_only_multilabel
import trainers.openclip_vptdeep_coop_multilabel_cached
import trainers.openclip_vptdeep_residual_multilabel_cached
import trainers.openclip_vptdeep_visualresidual_multilabel
import trainers.zsclip
import evaluators.multilabel


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX = 8
    cfg.TRAINER.VPT.N_DEEP_LAYERS = -1
    cfg.TRAINER.VPT.PREC = "amp"
    cfg.TRAINER.VPT.INIT_STD = 0.02
    cfg.TRAINER.VPT.DROPOUT = 0.0
    cfg.TRAINER.VPT.TRAIN_LOGIT_SCALE = False
    cfg.TRAINER.VPT.TRAIN_PROJ = False
    cfg.TRAINER.VPT.TRAIN_NORM = False
    cfg.TRAINER.VPT.LOSS_NAME = "bce"
    cfg.TRAINER.VPT.ASL_GAMMA_NEG = 4.0
    cfg.TRAINER.VPT.ASL_GAMMA_POS = 1.0
    cfg.TRAINER.VPT.ASL_CLIP = 0.05
    cfg.TRAINER.VPT.ASL_EPS = 1e-8
    cfg.TRAINER.VPT.FROZEN_MODEL_PATH = ""

    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9
    cfg.TRAINER.MAPLE.PREC = "amp"
    cfg.TRAINER.MAPLE.INIT_STD = 0.02
    cfg.TRAINER.MAPLE.CROSS_ATTN_DIM = 512
    cfg.TRAINER.MAPLE.CROSS_ATTN_HEADS = 4

    cfg.TRAINER.REGION = CN()
    cfg.TRAINER.REGION.PREC = "amp"
    cfg.TRAINER.REGION.QUERY_TEMPLATE = "a fundus photograph of {}."
    cfg.TRAINER.REGION.ATTN_HEADS = 4
    cfg.TRAINER.REGION.GATE_INIT_BIAS = -2.0

    cfg.TRAINER.STAGE2 = CN()
    cfg.TRAINER.STAGE2.MODE = "affine"
    cfg.TRAINER.STAGE2.RANK = 16
    cfg.TRAINER.STAGE2.HIDDEN_DIM = 512
    cfg.TRAINER.STAGE2.DROPOUT = 0.0
    cfg.TRAINER.STAGE2.PREC = "amp"
    cfg.TRAINER.STAGE2.TEXT_TEMPLATE = "a fundus photograph of {}."
    cfg.TRAINER.STAGE2.CORR_HEADS = 4
    cfg.TRAINER.STAGE2.CORR_LAYERS = 2
    cfg.TRAINER.STAGE2.TIP_ALPHA_INIT = 0.1
    cfg.TRAINER.STAGE2.TIP_BETA_INIT = 5.5
    cfg.TRAINER.STAGE2.TAIL_POWER = 0.5
    cfg.TRAINER.STAGE2.TAIL_THRESHOLD = 20
    cfg.TRAINER.STAGE2.HEAD_THRESHOLD = 100
    cfg.TRAINER.STAGE2.HARDNEG_THRESHOLD = 0.4
    cfg.TRAINER.STAGE2.HARDNEG_ALPHA = 4.0
    cfg.TRAINER.STAGE2.HARDNEG_GAMMA = 2.0
    cfg.TRAINER.STAGE2.CANDIDATE_THRESHOLD = 0.3
    cfg.TRAINER.STAGE2.CANDIDATE_TOPK = 6
    cfg.TRAINER.STAGE2.CANDIDATE_SOFT_SCALE = 12.0
    cfg.TRAINER.STAGE2.RECALL_HELPER_ALPHA = 0.75
    cfg.TRAINER.STAGE2.HARDCLASS_USE_MID = True

    cfg.TRAINER.STAGE2VIS = CN()
    cfg.TRAINER.STAGE2VIS.MODE = "cnn_logit"
    cfg.TRAINER.STAGE2VIS.HIDDEN_DIM = 256
    cfg.TRAINER.STAGE2VIS.ALPHA_INIT = 0.1
    cfg.TRAINER.STAGE2VIS.PATCH_RATIO = 0.6
    cfg.TRAINER.STAGE2VIS.PREC = "amp"
    cfg.TRAINER.STAGE2VIS.CANDIDATE_THRESHOLD = 0.3
    cfg.TRAINER.STAGE2VIS.CANDIDATE_TOPK = 6
    cfg.TRAINER.STAGE2VIS.CANDIDATE_SOFT_SCALE = 12.0
    cfg.TRAINER.STAGE2VIS.TEACHER_CKPT = ""
    cfg.TRAINER.STAGE2VIS.DISTILL_LAMBDA = 0.5
    cfg.TRAINER.STAGE2VIS.DISTILL_TEMPERATURE = 1.0
    cfg.TRAINER.STAGE2VIS.INIT_FROM_TEACHER = False
    cfg.TRAINER.STAGE2VIS.INCEPTION_PRETRAINED = True
    cfg.TRAINER.STAGE2VIS.RELEASE_INFERENCE = False
    cfg.TRAINER.STAGE2VIS.HARDNEG_THRESHOLD = 0.4
    cfg.TRAINER.STAGE2VIS.HARDNEG_ALPHA = 4.0
    cfg.TRAINER.STAGE2VIS.HARDNEG_GAMMA = 2.0
    cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_RATIO = 0.55
    cfg.TRAINER.STAGE2VIS.LOCAL_PATCH_WEIGHT = 0.75
    cfg.TRAINER.STAGE2VIS.CLASSSPEC_TEMP = 8.0
    cfg.TRAINER.STAGE2VIS.CASCADE_LOW_THRESHOLD = 0.2
    cfg.TRAINER.STAGE2VIS.CASCADE_HIGH_THRESHOLD = 0.6
    cfg.TRAINER.STAGE2VIS.CONFUSION_TOPK = 6
    cfg.TRAINER.STAGE2VIS.CONFUSION_MIX = 1.5
    cfg.TRAINER.STAGE2VIS.ROI_TOPK = 4
    cfg.TRAINER.STAGE2VIS.ROI_RATIO_LARGE = 0.65
    cfg.TRAINER.STAGE2VIS.ROI_RATIO_MEDIUM = 0.45
    cfg.TRAINER.STAGE2VIS.ROI_RATIO_SMALL = 0.30
    cfg.TRAINER.STAGE2VIS.MEMORY_TOPK = 32
    cfg.TRAINER.STAGE2VIS.MEMORY_TEMP = 10.0
    cfg.TRAINER.STAGE2VIS.MEMORY_TAIL_THRESHOLD = 40
    cfg.TRAINER.STAGE2VIS.MULTI_PROTO_K = 4
    cfg.TRAINER.STAGE2VIS.CONFUSION_EXEMPLAR_TOPK = 16
    cfg.TRAINER.STAGE2VIS.RETRIEVAL_ALPHA_INIT = 0.05

    cfg.TRAINER.LORA = CN()
    cfg.TRAINER.LORA.RANK = 8
    cfg.TRAINER.LORA.ALPHA = 16
    cfg.TRAINER.LORA.DROPOUT = 0.0
    cfg.TRAINER.LORA.N_BLOCKS = 4
    cfg.TRAINER.LORA.PREC = "amp"
    cfg.TRAINER.LORA.ATTN_IN = True
    cfg.TRAINER.LORA.ATTN_OUT = True
    cfg.TRAINER.LORA.MLP_FC = True
    cfg.TRAINER.LORA.MLP_PROJ = True

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.MANIFEST_PATH = ""
    cfg.DATASET.LABEL_MAP_PATH = ""
    cfg.DATASET.IMAGE_CACHE_DIR = ""
    cfg.DATASET.IMAGE_CACHE_EXT = "jpg"
    cfg.DATASET.INCEPTION_IMAGE_CACHE_DIR = ""
    cfg.DATASET.MAX_TRAIN_SAMPLES = -1
    cfg.DATASET.MAX_VAL_SAMPLES = -1
    cfg.DATASET.MAX_TEST_SAMPLES = -1
    cfg.DATASET.FEATURE_CACHE_PATH = ""
    cfg.DATASET.FEATURE_CACHE_BATCH_SIZE = 64
    cfg.DATASET.FEATURE_CACHE_NUM_WORKERS = 4
    cfg.DATASET.FEATURE_CACHE_REBUILD = False


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
