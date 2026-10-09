"""Make console output survive a non-UTF-8 terminal.

On Windows the default console codepage is cp1252. A script that prints a label
containing ``>=`` as U+2265, or ``->`` as U+2192, raises ``UnicodeEncodeError``
there -- and the failure lands wherever the print happens, which is typically
*after* the script has already written its output file. The artifact is correct,
the run reports as failed, and the traceback points at a print statement rather
than anything meaningful.

``errors="replace"`` is the important half: a summary line should degrade to a
``?`` rather than take down a run that has already done its work.

Call once at the top of ``main()`` in any script that prints text it did not
construct itself -- labels read from JSON, intent names, partition tags.
"""

from __future__ import annotations

import sys


def safe_stdout() -> None:
    """Reconfigure stdout/stderr to UTF-8, replacing anything unencodable."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:  # absent when the stream is redirected
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass  # already detached, or not reconfigurable
