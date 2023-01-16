from dotmap import DotMap
from .constants import DEFAULT_CONFIG_PATH
from .core.file import load_json

DEFAULT_CONFIG = DotMap(load_json(DEFAULT_CONFIG_PATH))
