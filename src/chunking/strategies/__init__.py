import importlib
import pkgutil

# Auto-import all strategy modules so they can self-register
for m in pkgutil.iter_modules(__path__):
    if not m.ispkg:
        importlib.import_module(f"{__name__}.{m.name}")
