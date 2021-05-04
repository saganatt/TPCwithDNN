# pylint: disable=missing-module-docstring, missing-function-docstring

import sys
import resource
import psutil

from tpcwithdnn.logger import get_logger

def get_memory_usage(obj):
    return sys.getsizeof(obj)

def format_memory(size):
    if 1024 <= size < 1024**2:
        return size // 1024, 'k'
    if 1024**2 <= size < 1024**3:
        return size // (1024**2), 'M'
    if 1024**3 <= size:
        return size // (1024**3), 'G'
    return size, ''

def print_total_memory_usage():
    logger = get_logger()
    size, mult = format_memory(psutil.virtual_memory().available)
    logger.info("Free RAM: %d %sB", size, mult)
    size, mult = format_memory(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    logger.info("RAM used by application: %d %sB", size, mult)
