"""Console fixes for GUI-launched training processes (auto-loaded by Python).

The GUI pipes a training process's output so it can keep the last lines for
error reports, then copies everything to its own console. On a pipe, the
training scripts behave as if they had no terminal. Python imports this
module at startup because CommandExecutor puts this folder first on
PYTHONPATH, only when the GUI itself runs in a real terminal and only for its
own child processes (``KOHYA_CHILD_CONSOLE=1``). It:

* turns off rich's legacy-Windows mode, which colours through the Win32
  console API and therefore drops all colour on a pipe;
* shows the timestamp on every log line (rich collapses repeated times);
* starts a new line before any output that arrives while a ``\\r`` progress
  bar (tqdm) is mid-line, so log lines never get glued onto the bar.

Width, forced colour and unbuffered stdout come from environment variables
set alongside (COLUMNS, FORCE_COLOR, TQDM_NCOLS, PYTHONUNBUFFERED).
Every step is best-effort: a failure here must never stop training.
"""

import os
import sys


class _LineState:
    """Shared by stdout and stderr: both go to the same pipe."""

    progress_open = False


class _ProgressAwareStream:
    """Insert "\\n" before foreign output while a "\\r" progress line is open."""

    def __init__(self, stream, state):
        self._stream = stream
        self._state = state

    def write(self, s):
        if not s:
            return self._stream.write(s)
        state = self._state
        # "\r" redraws the bar; a leading "\n" (e.g. tqdm closing) ends it.
        if state.progress_open and not s.startswith(("\r", "\n")):
            self._stream.write("\n")
            state.progress_open = False
        n = self._stream.write(s)
        last_nl = s.rfind("\n")
        last_cr = s.rfind("\r")
        if last_cr > last_nl:
            state.progress_open = True  # a bar was (re)drawn and is still open
        elif last_nl != -1:
            state.progress_open = False
        return n

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install():
    state = _LineState()
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is not None and not isinstance(stream, _ProgressAwareStream):
            setattr(sys, name, _ProgressAwareStream(stream, state))

    try:
        import rich.console
        import rich.logging

        # Colours as ANSI codes (which pass through the pipe), not Win32 calls.
        rich.console.detect_legacy_windows = lambda: False

        original_init = rich.logging.RichHandler.__init__
        if not getattr(original_init, "_kohya_child_console", False):

            def __init__(self, *args, **kwargs):
                kwargs.setdefault("omit_repeated_times", False)
                original_init(self, *args, **kwargs)

            __init__._kohya_child_console = True
            rich.logging.RichHandler.__init__ = __init__
    except Exception:
        pass


if os.environ.get("KOHYA_CHILD_CONSOLE") == "1":
    try:
        _install()
    except Exception:
        pass
