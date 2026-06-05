"""공통 로거 설정."""

import logging
import sys

_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
