import yaml


DEFAULTS = {
    "init_rand_seed": 18681265445,  # random seed for reproducibility, a large number is preferred
    "dataset_name": "epic",  # dataset loader, specify the dataset here
    "devices": ['cuda:0'],  # default: single gpu
    "train_split": ('training', ),
    "val_split": ('validation', ),
    "model_name": "LocPointTransformer",
    "dataset": {
        "feat_stride": 16,  # temporal stride of the feats
        "num_frames": 32,  # number of frames for each feat
        "default_fps": None,  # default fps, may vary across datasets; Set to none for read from json file
        "input_dim": 2304,  # input feat dim
        "num_classes": 97,  # number of classes
        "downsample_rate": 1,  # downsampling rate of features, 1 to use original resolution
        "max_seq_len": 2304,  # max sequence length during training
        "trunc_thresh": 0.5,  # threshold for truncating an action
        "crop_ratio": None,  # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        "force_upsampling": False,  # if true, force upsampling of the input features into a fixed size
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    "model": {
        "backbone_type": 'conv',  # type of backbone (convTransformer | conv | mlp )
        "fpn_type": "identity",  # type of FPN (fpn | identity)
        "backbone_arch": (1, 2, 7),
        "scale_factor": 2,  # scale factor between pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 10000)],  # regression range for pyramid levels
        "n_head": 4,  # number of heads in self-attention
        "n_mha_win_size": -1,  # window size for self attention; <=1 to use full seq (ie global attention)
        "embd_kernel_size": 3,  # kernel size for embedding network
        "embd_dim": 512,  # (output) feature dim for embedding network
        "embd_with_ln": True,  # if attach group norm to embedding network
        "fpn_dim": 512,  # feat dim for FPN
        "fpn_with_ln": True,  # if add ln at the end of fpn outputs
        "head_dim": 512,  # feat dim for head
        "head_kernel_size": 3,  # kernel size for reg/cls/center heads
        "head_with_ln": True,  # if attach group norm to heads
        "max_buffer_len_factor": 6.0,  # defines the max length of the buffered points
        "use_abs_pe": False,  # disable abs position encoding (added to input embedding)
        "use_rel_pe": False,  # use rel position encoding (added to self-attention)
    },
    "train_cfg": {
        "center_sample": "radius",  # radius | none (if to use center sampling)
        "center_sample_radius": 1.5,
        "loss_weight": 1.0,  # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        "clip_grad_l2norm": -1,  # gradient cliping, not needed for pre-LN transformer
        "head_empty_cls": [],  # cls head without data (a fix to epic-kitchens / thumos)
        "dropout": 0.1,  # dropout ratios for tranformers
        "droppath": 0.1,  # ratio for drop path
        "label_smoothing": 0.0,  # if to use label smoothing (>0.0)
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft',  # soft | hard | none
        "nms_sigma": 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh": 0.75,
    },
    "opt": {
        "type": "AdamW",  # optimizer type: SGD or AdamW
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        "epochs": 30,  # excluding the warmup epochs
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",  # lr scheduler: cosine / multistep
        "schedule_steps": [],  # in #epochs excluding warmup
        "schedule_gamma": 0.1,
    }
}


def _merge(src, dst):
    """
    Recursively merge src dictionary into dst dictionary.
    """
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v


def load_default_config():
    """
    Load the default configuration.
    """
    config = DEFAULTS
    return config


def _update_config(config):
    """
    Update the configuration with derived fields.
    """
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config


def load_config(config_file, defaults=DEFAULTS):
    """
    Load the configuration from a YAML file and merge it with the default configuration.
    """
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config
