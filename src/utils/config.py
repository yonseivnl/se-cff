from yacs.config import CfgNode as CN


def get_cfg(cfg_path):
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    return cfg
