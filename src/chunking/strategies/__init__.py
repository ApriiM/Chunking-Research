import importlib
import pkgutil
import warnings

# Auto-import all strategy modules so they can self-register
for m in pkgutil.iter_modules(__path__):
    if not m.ispkg:
        module_name = f"{__name__}.{m.name}"
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"Skipping chunking strategy '{module_name}' because dependency '{exc.name}' is missing.",
                RuntimeWarning,
            )
