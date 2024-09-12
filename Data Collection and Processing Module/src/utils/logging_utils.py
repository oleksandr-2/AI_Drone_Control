import logging
import logging.config
import yaml
import os

class LoggingUtils:
    @staticmethod
    def setup_logging(default_path='config/logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
        """
        Setup logging configuration.

        :param default_path: Path to the logging configuration file.
        :param default_level: Default logging level if no configuration file is found.
        :param env_key: Environment variable for the logging configuration file path.
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value

        if os.path.exists(path):
            with open(path, 'r') as f:
                config = yaml.safe_load(f.read())
                logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
            print(f'Logging configuration file not found. Using default logging level: {logging.getLevelName(default_level)}')

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance.

        :param name: Name of the logger.
        :return: Logger instance.
        """
        return logging.getLogger(name)

if __name__ == "__main__":
    # Example usage
    LoggingUtils.setup_logging()

    logger = LoggingUtils.get_logger(__name__)
    logger.info("Logging setup complete.")
    logger.debug("This is a debug message.")
    logger.error("This is an error message.")
