# -*- coding: utf-8 -*-
"""日志配置（设计报告 §8 common/logger.py）"""
import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """初始化根 logger（幂等）：输出到 stdout，统一格式"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取带统一格式的 logger（首次调用自动初始化）"""
    setup_logging()
    return logging.getLogger(name)
