import os
import yaml
from copy import deepcopy
from typing import Dict


def _defaults_path() -> str:
    '''
    Get the path to the chunker defaults YAML file.
    
    :return: Path to the chunker defaults YAML file
    :rtype: str
    '''
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "chunkers", "defaults.yaml")
    )


def _load_defaults_from_yaml() -> Dict[str, Dict[str, object]]:
    '''
    Load the chunker defaults from the YAML file.
    
    :return: Nested dictionary of chunker defaults
    :rtype: Dict[str, Dict[str, object]]
    ''' 
    path = _defaults_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunker defaults file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Chunker defaults file must map names to dicts: {path}")
    # Ensure nested dicts
    cleaned = {}
    for key, val in data.items():
        if isinstance(val, dict):
            cleaned[key] = val
    return cleaned


CHUNKER_DEFAULTS: Dict[str, Dict[str, object]] = _load_defaults_from_yaml()


def merge_with_defaults(name: str, overrides: Dict[str, object]) -> Dict[str, object]:
    '''
    Merge provided configuration overrides with the defaults for the given chunker name.
    
    :param name: Name of the chunker
    :type name: str
    :param overrides: Configuration overrides
    :type overrides: Dict[str, object]
    :return: Merged configuration dictionary
    :rtype: Dict[str, object]
    '''
    base = deepcopy(CHUNKER_DEFAULTS.get(name, {}))
    if overrides:
        base.update(overrides)
    return base
