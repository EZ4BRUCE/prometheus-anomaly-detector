"""docstring for packages."""

import time
import logging
from multiprocessing import Pool, Process, Queue
import tornado.ioloop
import tornado.web

from configuration import Configuration

from engine.server.server import make_app
import schedule

from engine.manager import AnalyzeManager

from pprint import pformat


if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    manager = AnalyzeManager(logger)

    # Set up the tornado web app
    app = make_app(logger, manager)
    app.listen(8789)
    server_process = Process(target=tornado.ioloop.IOLoop.instance().start)
    # Start up the server to expose the metrics.
    server_process.start()

    logger.info(
        "Will retrain model every %s minutes", Configuration.retraining_interval_minutes
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

    # join the server process in case the main process ends
    server_process.join()
