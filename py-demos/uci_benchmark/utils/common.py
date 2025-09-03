"""Common utilities for UCI benchmark."""

import os
import sys
import ctypes
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
