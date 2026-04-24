# 日志处理
# 负责记录日志方便调试和问题排查
import logging
import os
import textwrap
from datetime import datetime


class WrapFormatter(logging.Formatter):
    """自定义 Formatter：自动对超长 message 进行折行处理"""

    def __init__(self, fmt=None, datefmt=None, max_line_length=120, wrap_width=120):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.max_line_length = max_line_length
        self.wrap_width = wrap_width

    def format(self, record):
        original_message = record.getMessage()
        if len(original_message) > self.max_line_length:
            record.msg = textwrap.fill(
                original_message,
                width=self.wrap_width,
                subsequent_indent='                             ',
            )
        return super().format(record)

# Logger 采用单例模式：同一个名字返回同一个实例，避免重复添加 Handler
def get_logger(name='app', log_dir='logs', level=logging.DEBUG):
    """
    创建并返回一个记录到文件的 Logger 实例。

    参数:
        name:   Logger 名称（通常用 __name__ 传入当前模块名）
        log_dir: 日志文件存放目录
        level:  日志级别（DEBUG / INFO / WARNING / ERROR / CRITICAL）
    """
    logger = logging.getLogger(name)

    # 如果已存在 Handler，直接返回（防止重复添加）
    if logger.handlers:
        return logger

    # 设置日志级别：低于该级别的日志将被忽略
    logger.setLevel(level)

    # 创建文件 Handler：将日志写入文件
    os.makedirs(log_dir, exist_ok=True)
    # 按日期生成日志文件名，例如 2026-04-24.log
    log_file = os.path.join(log_dir, f'{datetime.now().strftime("%Y-%m-%d")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)

    # 创建格式化器：自定义折行格式，超过 300 字符自动换行
    formatter = WrapFormatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        max_line_length=300,
        wrap_width=300,
    )
    file_handler.setFormatter(formatter)

    # 将 Handler 绑定到 Logger
    logger.addHandler(file_handler)

    return logger