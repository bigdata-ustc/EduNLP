

import os
import logging
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

def get_logger(logfile):
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    logger = logging.getLogger(name="QuesQuality")
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(filename=logfile, encoding="utf-8", mode="w")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    
    consolehandler = logging.StreamHandler()
    consolehandler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(consolehandler)   # log to file and print to console
    return logger


def get_pk(y_pred, y, k):
    tag_num = len(y)
    count = 0
    for i in range(0, tag_num-k):
        seg_count_y_pred = 0
        seg_count_y = 0
        for j in range(i, i+k):
            seg_count_y_pred += y_pred[j]
            seg_count_y += y[j]
        if seg_count_y_pred != seg_count_y:
            count += 1
    return count
