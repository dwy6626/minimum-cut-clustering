import logging
from sys import stdout


__module_logger = logging.getLogger('minimum-cut clustering')
__module_logger.addHandler(logging.StreamHandler(stdout))
__module_logger.setLevel(logging.INFO)


def print_more(msg, *args, **kwargs):
    __module_logger.debug(msg, *args, **kwargs)


def print_normal(msg, *args, **kwargs):
    __module_logger.info(msg, *args, **kwargs)


def set_module_logger(log_mode):
    __module_logger.info('set logger level to {}'.format(log_mode))
    __module_logger.setLevel(log_mode)


def get_module_logger_level():
    return __module_logger.getEffectiveLevel()


# Print many many stars
def print_1_line_stars():
    print_normal('')
    print_normal('*' * 72)
    print_normal('')
