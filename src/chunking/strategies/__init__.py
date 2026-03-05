import importlib
import pkgutil
import warnings

# Auto-import all strategy modules so they can self-register
for m in pkgutil.iter_modules(__path__):
    if not m.ispkg:
        try:
            importlib.import_module(f"{__name__}.{m.name}")
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"Skipping strategy module '{m.name}' due to missing optional dependency '{exc.name}'.",
                RuntimeWarning,
            )
