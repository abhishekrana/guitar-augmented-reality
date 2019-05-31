import os
import logging
import datetime


def logger_init(output_dir, log_level):
    logging.disabled = False


    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    log_file_path_name = os.path.join(output_dir, 'log_' + timestamp)
    logFormatter = logging.Formatter("%(asctime)s | %(filename)20s:%(lineno)s | %(funcName)20s() | %(message)s")

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_path_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(log_level)

    logging.getLogger('PIL').setLevel(logging.ERROR)

    return fileHandler, consoleHandler


