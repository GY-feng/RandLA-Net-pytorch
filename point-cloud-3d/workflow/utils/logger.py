import logging
import sys
from datetime import datetime
from pathlib import Path

# 定义颜色代码
class LogColors:
    DEBUG = '\033[0m'  # 默认
    INFO = '\033[092m'    # 绿色
    WARNING = '\033[93m' # 黄色
    ERROR = '\033[91m'   # 红色
    RESET = '\033[0m'    # 重置颜色

# 自定义Formatter，添加颜色和格式
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # 获取调用日志的文件名（不含路径和扩展名）
        caller_path = Path(record.pathname)
        caller_info = f"{caller_path.stem}"
        
        # 设置颜色
        color = {
            logging.DEBUG: LogColors.DEBUG,
            logging.INFO: LogColors.INFO,
            logging.WARNING: LogColors.WARNING,
            logging.ERROR: LogColors.ERROR,
        }.get(record.levelno, LogColors.INFO)
        
        # 格式化日志
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = super().format(record)
        return f"{timestamp} - {color}{caller_info} - {message}{LogColors.RESET}"

# 配置日志系统
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 设置最低日志级别

    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    formatter = ColoredFormatter(
        '%(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger

# 创建全局logger实例
logger = setup_logger()

def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

if __name__ == "__main__":
    debug("This is a debug message")
    info("This is an info message")
    warning("This is a warning message")
    error("This is an error message")