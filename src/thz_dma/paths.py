"""Read relative manifest paths written by either Windows or Linux."""
from pathlib import Path, PurePosixPath, PureWindowsPath


def relative_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if (not path.parts or path.is_absolute() or PureWindowsPath(value).drive
            or ".." in path.parts):
        raise ValueError("manifest path must be relative and remain inside its root")
    return Path(*path.parts)
