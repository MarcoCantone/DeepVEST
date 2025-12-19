from config import load_config
import os
import yaml

cfg_path = "/data/cantone/experiments/MRI/1-base/config.yaml"

cfg = load_config(cfg_path)

src_dir = "/data/cantone/experiments/MRI/3-var_test_size/"

runs = ["1", "2", "3", "4", "5"]
sizes = [0.1, 0.15, 0.2]

for run in runs:
    for size in sizes:
        dir_name = f"size{str(size)}_run{run}"
        # cfg["OPTIMIZER"]["params"]["lr"] = lr
        cfg["TRAINING"]["device"] = "cuda:1"
        cfg["TRAINING"]["train_test_split_seed"] = None
        cfg["TRAINING"]["test_size"] = size
        cfg["TRAINING"]["workspace"] = os.path.join(src_dir, dir_name)

        os.system(f'mkdir {os.path.join(src_dir, dir_name)}')
        with open(os.path.join(src_dir, dir_name, "config.yaml"), "w") as f:
            f.write(yaml.dump(cfg))


dirs = os.listdir(src_dir)

exp_script = "/home/cantone/PyCharm_sync/MRI/experiment_monai.py"

for dir in dirs:
    print(f'python3 -u {exp_script} --config {os.path.join(src_dir, dir, "config.yaml")} > {os.path.join(src_dir, dir, "log.txt")};')

