import inspect
import pkgutil
import importlib


class Processor:
    def __init__(self):
        self.description = "UNKNOWN"


class ProcessorManager:
    def __init__(self, processors_dir="api.augmentors"):
        self.processors_dir = processors_dir
        self.processors = {}
        self.load_processors()

    def load_processors(self):
        for importer, modname, ispkg in pkgutil.iter_modules(
            importlib.import_module(self.processors_dir).__path__
        ):
            if not ispkg:
                module = importlib.import_module(f"{self.processors_dir}.{modname}")
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Processor) and obj != Processor:
                        self.processors[obj.__name__] = obj
