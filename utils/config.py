import yaml
from pathlib import Path
import types
import os

def load_config(path):
    conf = yaml.safe_load(Path(path).read_text())
    cfg = types.SimpleNamespace(**conf)
    
    exps_old = [int(d.replace("exp", "")) for d in os.listdir(cfg.runs_dir) if d.startswith("exp")]
    cfg.exp_num = max(exps_old)+1 if len(exps_old) > 0 else 0
    cfg.exp_name = f"exp{cfg.exp_num}"
    cfg.save_dir = os.path.join(cfg.runs_dir, cfg.exp_name)
    cfg.demo_weights = os.path.join(cfg.runs_dir, cfg.demo_run, "last.pt") if cfg.demo_run is not None else None
    return cfg