import logging

logging.basicConfig()


def setup_logger(name="denoiseg", path=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if path is not None:
        formatter = logging.Formatter(
            "%(process)d: %(asctime)s - %(filename)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger
