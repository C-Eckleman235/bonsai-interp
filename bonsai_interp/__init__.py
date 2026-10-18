"""bonsai_interp: forward, inverse and steering probes for the neurons of Bonsai-8B, and the interp graph."""
import logging as _logging
import sys as _sys

# Windows consoles default to a legacy code page; corpus text and the ⟦peak⟧ markers need UTF-8.
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
# torch prints a harmless "Redirects are currently not supported in Windows" warning on import.
_logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(_logging.ERROR)
