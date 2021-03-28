import importlib.util
from pathlib import Path

try:
    import git
    _git_available = True
except ImportError:  # can cause import errors, not because of package but because of git
    _git_available = False

import decode


def decode_state() -> str:
    """Get version tag of decode. If in repo this will get you the output of git describe.

    Returns git describe, decode version or decode version with invalid appended.
    """

    p = Path(importlib.util.find_spec('decode').origin).parents[1]

    if _git_available:
        try:
            r = git.Repo(p)
            return r.git.describe(dirty=True)

        except git.exc.InvalidGitRepositoryError:  # not a repo but an installed package
            return decode.__version__

        except git.exc.GitCommandError:
            return "vINVALID-recent-" + decode.__version__
    else:
        return "vINVALID-recent-" + decode.__version__


if __name__ == '__main__':
    v = decode_state()
    print(f"DECODE version: {v}")
