import logging
import os
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama for Windows compatibility
init(autoreset=True)

# --- Define custom SUCCESS level ---
SUCCESS_LEVEL = 25
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")

def success(self, message, *args, **kwargs):
    """Add success-level logging to the logger."""
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)

logging.Logger.success = success


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    COLORS = {
        logging.DEBUG: Style.DIM + Fore.WHITE,     # Debug: Dim white
        logging.INFO: Fore.YELLOW,                 # Info: Yellow
        SUCCESS_LEVEL: Fore.GREEN,                 # Success: Green
        logging.WARNING: Fore.BLUE,                # Warning: Blue
        logging.ERROR: Fore.MAGENTA,               # Error: Magenta (optional)
        logging.CRITICAL: Fore.RED + Style.BRIGHT, # Critical: Bright red
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


def setup_logger(name="app_logger", log_dir="logs", level=logging.DEBUG):
    """Creates and configures a logger that logs messages to both console and a file."""
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # File handler (no color)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (with color)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    color_formatter = ColorFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(color_formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Example usage
if __name__ == "__main__":
    logger = setup_logger("colored_logger", level=logging.DEBUG)
    logger.debug("This is a debug message.")
    logger.info("This is an info message (yellow).")
    logger.success("This is a success message (green).")
    logger.warning("This is a warning message (blue).")
    logger.error("This is an error message (magenta).")
    logger.critical("This is a critical message (red).")
