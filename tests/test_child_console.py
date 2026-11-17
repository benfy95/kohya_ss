"""Console fixes for piped training processes:
kohya_gui/child_console/sitecustomize.py and the executor's env/line state."""

import importlib.util
import io
import os
import unittest
from pathlib import Path

from kohya_gui import class_command_executor as ce

_HOOK = Path(ce._CHILD_CONSOLE_DIR) / "sitecustomize.py"
_spec = importlib.util.spec_from_file_location("kohya_child_console", _HOOK)
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)  # KOHYA_CHILD_CONSOLE unset: installs nothing


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class TestProgressAwareStream(unittest.TestCase):
    def _streams(self):
        sink = io.StringIO()
        state = hook._LineState()
        out = hook._ProgressAwareStream(sink, state)
        err = hook._ProgressAwareStream(sink, state)
        return sink, out, err

    def test_log_line_after_open_bar_starts_on_new_line(self):
        sink, out, err = self._streams()
        err.write("\rsteps:  1%| | 1/10")
        err.write("\rsteps:  2%| | 2/10")
        err.write("INFO epoch is incremented\n")
        out.write("saving checkpoint: x\n")
        self.assertEqual(
            sink.getvalue(),
            "\rsteps:  1%| | 1/10\rsteps:  2%| | 2/10\n"
            "INFO epoch is incremented\nsaving checkpoint: x\n",
        )

    def test_bar_close_newline_is_not_doubled(self):
        sink, _, err = self._streams()
        err.write("\r100%| | 22/22")
        err.write("\n")
        err.write("INFO next\n")
        self.assertEqual(sink.getvalue(), "\r100%| | 22/22\nINFO next\n")

    def test_delegates_attributes(self):
        sink, out, _ = self._streams()
        self.assertFalse(out.isatty())
        out.flush()


class TestChildEnv(unittest.TestCase):
    def test_tty_env_adds_console_fixes(self):
        env = ce.console_env_for_child({"PYTHONPATH": "C:/x"}, console=FakeTTY())
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")
        self.assertEqual(env["KOHYA_CHILD_CONSOLE"], "1")
        self.assertTrue(int(env["TQDM_NCOLS"]) >= 40)
        first, rest = env["PYTHONPATH"].split(os.pathsep, 1)
        self.assertEqual(first, ce._CHILD_CONSOLE_DIR)
        self.assertEqual(rest, "C:/x")
        self.assertTrue(_HOOK.is_file())


class TestConsoleLineState(unittest.TestCase):
    def test_drain_tracks_mid_line(self):
        import collections

        state = {"mid_line": False}
        ce.drain_stream_to_buffer(
            io.StringIO("log\n\rsteps 5%"),
            collections.deque(maxlen=5),
            dest=io.StringIO(),
            console_state=state,
        )
        self.assertTrue(state["mid_line"])


if __name__ == "__main__":
    unittest.main()
