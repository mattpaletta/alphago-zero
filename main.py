import logging
import sys


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.NOTSET)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)s]')
    ch.setFormatter(formatter)
    root.addHandler(ch)

if __name__ == "__main__":
    setup_logging()
    logging.info("Learning Go!")