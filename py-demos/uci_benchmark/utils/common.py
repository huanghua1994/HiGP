"""Common utilities."""

import os
import sys
import ctypes
import tempfile
import stat
from contextlib import contextmanager


@contextmanager
def suppress_output():
    """Context manager to suppress stdout/stderr from C extensions."""
    try:
        libc = ctypes.CDLL(None)
        libc.fflush(None)
    except (OSError, AttributeError):
        pass

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()

        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    finally:
        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except (OSError, AttributeError):
            pass

        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        sys.stdout.flush()
        sys.stderr.flush()


@contextmanager
def capture_output():
    """Context manager to capture stdout/stderr from C extensions."""
    tmp_fd, tmp_filename = tempfile.mkstemp(prefix="higp_capture_", suffix=".txt")
    os.chmod(tmp_filename, stat.S_IRUSR | stat.S_IWUSR)

    try:
        libc = ctypes.CDLL(None)
        libc.fflush(None)
    except (OSError, AttributeError):
        pass

    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(tmp_fd, 1)
        os.dup2(tmp_fd, 2)
        yield tmp_filename
    finally:
        try:
            libc = ctypes.CDLL(None)
            libc.fflush(None)
        except (OSError, AttributeError):
            pass

        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

        if tmp_fd >= 0:
            try:
                os.close(tmp_fd)
            except OSError:
                pass

        sys.stdout.flush()
        sys.stderr.flush()
