import logging
import os
import warnings


class Logger:
    def __init__(self, log_file_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Capture all levels

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # FileHandler for logging to a file
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s-%(levelname)s-%(message)s',
            datefmt='%Y:%m:%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # # StreamHandler for console output
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)  # Set to INFO or WARNING for less verbosity
        # console_formatter = logging.Formatter(
        #     '%(name)s - %(levelname)s - %(message)s'
        # )
        # console_handler.setFormatter(console_formatter)
        # self.logger.addHandler(console_handler)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Redirect warnings to the logger
        logging.captureWarnings(True)
        # self.logger.warning = self.logger.warning

        # This assigns the warning handler
        warnings.showwarning = self._log_warning

    def _log_warning(self, message, category, filename, lineno, file=None, line=None):
        self.logger.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

    def log_and_print(self, message: str, level):
        self.logger.log(level=level, msg=message)
        print(message)

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def warning(self, message: str):
        self.logger.warning(message)