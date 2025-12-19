import yaml
import importlib

def load_config(config_file=None):
    if config_file is None:
        return cfg
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_config(config, path):
    with open(path, 'w') as file:
        outputs = yaml.dump(config, file)
        return outputs


def get_class_by_path(path):
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def create_object_from_dict_old(dict):
    cls = get_class_by_path(dict["class"])
    return cls(**dict["params"])


def create_object_from_dict(dict):
    cls = get_class_by_path(dict["class"])
    for key, value in dict["params"].items():
        if type(value) == type({}):
            dict["params"][key] = create_object_from_dict(value)
    return cls(**dict["params"])
