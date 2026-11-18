import codecs
import collections
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Deque, Optional, TextIO, Tuple

import gradio as gr
import psutil

from .common_gui import output_message
from .custom_logging import setup_logging

# Set up logging
log = setup_logging()

# Folder holding the sitecustomize.py auto-loaded by piped training processes.
_CHILD_CONSOLE_DIR = os.path.join(os.path.dirname(__file__), "child_console")

# How many trailing log lines to attach to a failure summary.
_DEFAULT_LOG_TAIL_LINES = 40


def tail_lines(text: Optional[str], max_lines: int = _DEFAULT_LOG_TAIL_LINES) -> str:
    """Return the last ``max_lines`` lines of ``text`` (empty-safe)."""
    if not text:
        return ""
    lines = text.splitlines()
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:])


def read_log_tail(path: str, max_lines: int = _DEFAULT_LOG_TAIL_LINES) -> str:
    """Read the last ``max_lines`` lines from a log file; empty if missing."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return tail_lines(f.read(), max_lines=max_lines)
    except OSError:
        return ""


# CSI / OSC escape sequences (colours, cursor moves) emitted by rich / tqdm.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences so failure summaries stay readable."""
    return _ANSI_ESCAPE_RE.sub("", text)


def drain_stream_to_buffer(
    stream,
    buffer: Deque[str],
    *,
    dest: Optional[TextIO] = None,
    chunk_size: int = 4096,
    console_state: Optional[dict] = None,
) -> None:
    """Copy ``stream`` to ``dest`` verbatim and keep its lines in ``buffer``.

    Used to tee subprocess output so a short failure tail is available without
    hiding live console logs. Output is forwarded in chunks (not lines) so
    carriage-return progress bars (tqdm) keep updating in place and colour
    codes reach the console unchanged. ``stream`` may be binary (decoded as
    UTF-8) or text. Lines in ``buffer`` are split on ``\\n``/``\\r`` and have
    ANSI codes removed. Safe to call on a background thread.

    ``console_state["mid_line"]`` records whether the console cursor was left
    mid-line (e.g. an open progress bar), so the GUI can end that line before
    logging its own messages.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    read = getattr(stream, "read1", None) or stream.read
    pending = ""

    def _keep(parts):
        for part in parts:
            part = strip_ansi(part).rstrip()
            if part:
                buffer.append(part)

    try:
        while True:
            chunk = read(chunk_size)
            if not chunk:
                break
            text = decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
            if dest is not None and text:
                try:
                    dest.write(text)
                    dest.flush()
                except (OSError, ValueError):
                    pass
                if console_state is not None:
                    console_state["mid_line"] = not text.endswith("\n")
            parts = re.split(r"\r\n|\r|\n", pending + text)
            pending = parts.pop()
            _keep(parts)
        tail = decoder.decode(b"", final=True)
        _keep([pending + tail])
    except (OSError, ValueError):
        # Stream closed underfoot (process exit) — ignore.
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def console_env_for_child(
    env: Optional[dict], console: Optional[TextIO] = None
) -> Optional[dict]:
    """Tell a piped child process it is still printing to this console.

    Piping the child's output (for the failure tail) makes rich see a non-TTY,
    so it drops colours and wraps at 80 columns. When the GUI itself runs in a
    real terminal, pass the terminal's width and force colour so the training
    log looks like the GUI's own log. User-set values win (setdefault).
    """
    console = console if console is not None else sys.stdout
    try:
        is_tty = console.isatty()
    except (AttributeError, ValueError, OSError):
        is_tty = False
    if not is_tty:
        return env
    env = dict(os.environ if env is None else env)
    columns = shutil.get_terminal_size().columns
    env.setdefault("FORCE_COLOR", "1")
    env.setdefault("TTY_COMPATIBLE", "1")
    env.setdefault("COLUMNS", str(columns))
    # One-line tqdm bars spanning the window (one column spare so the bar
    # never wraps, which would break its "\r" redraw).
    env.setdefault("TQDM_NCOLS", str(max(columns - 1, 40)))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # A pipe makes stdout block-buffered: print()-ed messages such as
    # "saving checkpoint: ..." would appear late (or be lost on Stop).
    env.setdefault("PYTHONUNBUFFERED", "1")
    # child_console/sitecustomize.py: colour on Windows pipes, a timestamp on
    # every log line, log lines never glued onto a progress bar.
    env["KOHYA_CHILD_CONSOLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (_CHILD_CONSOLE_DIR, env.get("PYTHONPATH", "")) if p
    )
    return env


def summarize_training_end(
    returncode: Optional[int],
    *,
    user_stopped: bool = False,
    log_tail: str = "",
) -> Tuple[str, str]:
    """Build a (level, message) pair for end-of-training status.

    Levels: ``info`` or ``error``. User-initiated stop is always reported
    as cancelled (info), even when the OS exit code is non-zero.
    """
    if user_stopped:
        return "info", "Training was stopped by the user."

    if returncode is None:
        return "info", "Training has ended."

    if returncode == 0:
        return "info", "Training has ended successfully (exit code 0)."

    msg = f"Training failed with exit code {returncode}."
    if log_tail and log_tail.strip():
        msg = f"{msg}\n\nLast log lines:\n{log_tail.strip()}"
    return "error", msg


class CommandExecutor:
    """
    A class to execute and manage commands.
    """

    def __init__(self, headless: bool = False):
        """
        Initialize the CommandExecutor.
        """
        self.headless = headless
        self.process = None
        self._stopped_by_user = False
        self._output_lines: Deque[str] = collections.deque(
            maxlen=_DEFAULT_LOG_TAIL_LINES
        )
        self._tee_thread: Optional[threading.Thread] = None
        # Whether teed output left the console cursor mid-line (open bar).
        self._console_state = {"mid_line": False}
        # Optional callable run after a successful or user-stopped run.
        self._on_finish = None

        with gr.Row():
            self.button_run = gr.Button("Start training", variant="primary")

            self.button_stop_training = gr.Button(
                "Stop training",
                visible=self.process is not None or headless,
                variant="stop",
            )

    def execute_command(self, run_cmd: str, on_finish=None, **kwargs):
        """
        Execute a command if no other command is currently running.

        Parameters:
        - run_cmd (str): The command to execute.
        - on_finish (callable, optional): called with no arguments once the
          command ends successfully or is stopped by the user (e.g. to
          post-process the files it saved). Not called on failure.
        - **kwargs: Additional keyword arguments to pass to subprocess.Popen.
        """
        if self.process and self.process.poll() is None:
            log.info("The command is already running. Please wait for it to finish.")
        else:
            # Reconstruct the safe command string for display
            command_to_run = " ".join(run_cmd)
            log.info(f"Executing command: {command_to_run}")

            self._stopped_by_user = False
            self._output_lines = collections.deque(maxlen=_DEFAULT_LOG_TAIL_LINES)
            self._tee_thread = None
            self._console_state = {"mid_line": False}
            self._on_finish = on_finish

            # Tee child stdout/stderr into a ring buffer (and the console) so
            # failure summaries can include a traceback tail. Skip if the
            # caller already redirected either stream.
            tee_output = "stdout" not in kwargs and "stderr" not in kwargs
            popen_kwargs = dict(kwargs)
            if tee_output:
                popen_kwargs["stdout"] = subprocess.PIPE
                popen_kwargs["stderr"] = subprocess.STDOUT
                # Binary pipe: drain_stream_to_buffer decodes UTF-8 itself and
                # forwards "\r" untouched (text mode would turn every tqdm
                # refresh into a new line).
                popen_kwargs["bufsize"] = 0
                popen_kwargs["env"] = console_env_for_child(popen_kwargs.get("env"))

            # Execute the command securely
            self.process = subprocess.Popen(run_cmd, **popen_kwargs)
            log.debug("Command executed.")

            if tee_output and self.process.stdout is not None:
                self._tee_thread = threading.Thread(
                    target=drain_stream_to_buffer,
                    args=(self.process.stdout, self._output_lines),
                    kwargs={
                        "dest": sys.stdout,
                        "console_state": self._console_state,
                    },
                    name="command-executor-tee",
                    daemon=True,
                )
                self._tee_thread.start()

    def kill_command(self):
        """
        Kill the currently running command and its child processes.
        """
        if self.is_running():
            try:
                # Get the parent process and kill all its children
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                self._stopped_by_user = True
                self._join_tee(timeout=1.0)
                self._end_console_line()
                log.info("The running process has been terminated.")
            except psutil.NoSuchProcess:
                # Explicitly handle the case where the process does not exist
                log.info(
                    "The process does not exist. It might have terminated before the kill command was issued."
                )
            except Exception as e:
                # General exception handling for any other errors
                log.info(f"Error when terminating process: {e}")
        else:
            self.process = None
            log.info("There is no running process to kill.")

        return gr.Button(visible=True), gr.Button(visible=False or self.headless)

    def _end_console_line(self) -> None:
        """Finish a line the child left open (e.g. a progress bar) so the
        GUI's next log message starts on its own line."""
        if self._console_state.get("mid_line"):
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
            self._console_state["mid_line"] = False

    def _join_tee(self, timeout: float = 2.0) -> None:
        thread = self._tee_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _collect_log_tail(self) -> str:
        """Prefer teed process lines; fall back to GUI setup.log."""
        self._join_tee()
        if self._output_lines:
            return "\n".join(self._output_lines)
        # custom_logging writes the GUI log here (cwd-relative).
        return read_log_tail("setup.log", max_lines=_DEFAULT_LOG_TAIL_LINES)

    def wait_for_training_to_end(self):
        while self.is_running():
            time.sleep(1)
            log.debug("Waiting for training to end...")

        self._join_tee()
        self._end_console_line()

        returncode: Optional[int] = None
        if self.process is not None:
            # Ensure returncode is populated after the process has exited.
            returncode = self.process.poll()
            if returncode is None:
                returncode = self.process.returncode

        include_tail = returncode not in (None, 0) and not self._stopped_by_user
        log_tail = self._collect_log_tail() if include_tail else ""
        level, message = summarize_training_end(
            returncode,
            user_stopped=self._stopped_by_user,
            log_tail=log_tail,
        )

        if level == "error":
            log.error(message)
            output_message(msg=message, title="Training failed", headless=self.headless)
        else:
            log.info(message)

        on_finish, self._on_finish = self._on_finish, None
        if on_finish is not None and (returncode == 0 or self._stopped_by_user):
            try:
                on_finish()
            except Exception as e:  # post-processing must not break the UI
                log.error(f"Post-training step failed: {e}")

        return gr.Button(visible=True), gr.Button(visible=False or self.headless)

    def is_running(self):
        """
        Check if the command is currently running.

        Returns:
        - bool: True if the command is running, False otherwise.
        """
        return self.process is not None and self.process.poll() is None
