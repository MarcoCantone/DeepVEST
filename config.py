"""Configuration utilities.

Every experiment is described by a YAML file (see ``configs/``). Objects such as
transforms, networks, losses and metrics are written as::

    class: <python.path.to.Class>
    params: {<constructor keyword arguments>}

and instantiated with :func:`create_object_from_dict`.
"""

import importlib

import yaml


def load_config(config_file):
    """Load a YAML configuration file into a dictionary."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_config(config, path):
    """Write a configuration dictionary to a YAML file."""
    with open(path, 'w') as file:
        outputs = yaml.dump(config, file)
        return outputs


def get_class_by_path(path):
    """Return the class (or function) identified by a dotted path, e.g. ``monai.transforms.LoadImaged``."""
    module_path, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def create_object_from_dict(dict):
    """Instantiate ``dict["class"](**dict["params"])``.

    Parameters that are themselves dictionaries are interpreted as nested object
    specifications and instantiated recursively (e.g. the ``reader`` of ``LoadImaged``).

    Note: nested dictionaries are replaced in place by the created objects, so pass a
    copy of the configuration if it has to be reused afterwards.
    """
    cls = get_class_by_path(dict["class"])
    for key, value in dict["params"].items():
        if type(value) == type({}):
            dict["params"][key] = create_object_from_dict(value)
    return cls(**dict["params"])
