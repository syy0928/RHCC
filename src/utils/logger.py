import datetime
import logging

import numpy
import platform


class Logger:

    @staticmethod
    def divider(title):
        print(
            "\n============================================{}=============================================\n".format(
                title), flush=True)

    @staticmethod
    def info(message):
        time = datetime.datetime.now()
        timeStr = time.strftime("[%Y%m%d-%H:%M:%S]")
        print(timeStr + '=>[info]: {}'.format(message), flush=True)

    @staticmethod
    def getTimeStr(time):
        if platform.system().lower() != 'windows':
            timeStr = time.strftime("[%m%d-%H:%M:%S]")
        else:
            timeStr = time.strftime("[%m%d-%H-%M-%S]")
        return timeStr


def setup_basic_file_logger(log_file_path="app.log"):
    # 1. 获取日志器（可指定名称，避免与其他模块冲突）
    logger = logging.getLogger("MyAppLogger")
    logger.setLevel(logging.INFO)  # 全局日志级别：只记录 INFO 及以上（DEBUG 会被过滤）

    # 2. 避免重复添加处理器（多次调用函数时防止日志重复输出）
    if logger.handlers:
        return logger

    # 3. 配置「文件处理器」：指定输出文件和编码
    file_handler = logging.FileHandler(
        filename=log_file_path,
        mode="a",  # 追加模式（"w" 为覆盖模式，慎用）
        encoding="utf-8"  # 避免中文乱码
    )

    # 4. 配置日志格式（包含：时间、日志级别、日志器名称、内容）
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
    )
    file_handler.setFormatter(log_format)

    # 5. 将处理器添加到日志器
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    a = numpy.arange(1, 10, 1)
    Logger.info("test {}".format(a))
    time = datetime.datetime.now()
    print(Logger.getTimeStr(time))
