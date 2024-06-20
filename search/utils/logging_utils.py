import logging

# Create a custom formatter
class CustomFormatter(logging.Formatter):
    def format(self, record):
        log_message = f"{record.levelname}: {record.filename} - {record.funcName}() - Line {record.lineno}: {record.msg}"
        return log_message

# Create a logger
logger = logging.getLogger(__name__)

# Create a StreamHandler with the custom formatter
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomFormatter())

# Attach the StreamHandler to the logger
logger.addHandler(stream_handler)

# Set the log level
logger.setLevel(logging.INFO)
