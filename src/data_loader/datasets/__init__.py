"""Dataset-specific loaders live here and self-register via the registry."""

import importlib
import pkgutil

# auto-import all dataset modules so they can self-register
for m in pkgutil.iter_modules(__path__):
	if not m.ispkg:
		importlib.import_module(f"{__name__}.{m.name}")
