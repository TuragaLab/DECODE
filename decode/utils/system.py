import platform

import torch
import socket


def collect_system() -> dict:
    """collect system information for logging"""
    return {
        "hostname": socket.gethostname(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "alias": platform.platform(aliased=True)
        },
    }
