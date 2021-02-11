import logging
from rich.logging import RichHandler


def init_log():
    logging.basicConfig(filename='experiment_result.log',
                        filemode="w",
                        format='%(levelname)s\t%(message)s\t%(asctime)s\t%(pathname)s\tLine:%(lineno)d',
                        level=logging.DEBUG)
    handler = RichHandler()
    logging.getLogger('').addHandler(handler)
