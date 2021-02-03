import importlib.util
from pathlib import Path

import git

import decode


def decode_state() -> str:
    """Get version tag of decode. If in repo this will get you the output of git describe."""

    p = Path(importlib.util.find_spec('decode').origin).parents[1]

    try:
        r = git.Repo(p)
        return r.git.describe(dirty=True)

    except git.exc.InvalidGitRepositoryError:  # not a repo but an installed package
        return decode.__version__


if __name__ == '__main__':
    print(f"DECODE version: decode_state()")
