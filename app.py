"""docstring for packages."""

import time
import logging
from multiprocessing import Pool, Process, Queue
import tornado.ioloop
import tornado.web

from engine.server.server import make_app
import schedule

from engine.manager import AnalyzeManager

from pprint import pformat

import yaml
import argparse
import atexit
import requests

from logfmt_logger import getLogger


def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def init_analyzers(url: str, metrics: list[dict]):
    try:
        for metric in metrics:
            response = requests.post(
                url,
                json={
                    "metric": metric["metric"],
                    "model": metric["model"],
                    "window_size": metric["window_size"],
                    "sync_new_series_interval_seconds": metric[
                        "sync_new_series_interval_seconds"
                    ],
                    "retraining_interval_minutes": metric[
                        "retraining_interval_minutes"
                    ],
                },
            )
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"failed to init analyzers: {e}")


def main():
    # Set up logging
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    args = parser.parse_args()

    logger = getLogger("detection-engine")

    logger.info("Loading configuration from %s", args.config)
    config = load_config(args.config)
    logger.info("Configuration loaded:\n%s", pformat(config))

    manager = AnalyzeManager(logger, config["cluster_mode"], config["prometheus_url"])
    atexit.register(manager.cleanup)

    # Set up the tornado web app
    app = make_app(logger, manager)
    app.listen(config["server"]["port"])
    server_process = Process(target=tornado.ioloop.IOLoop.instance().start)
    # Start up the server to expose the metrics.
    server_process.start()

    time.sleep(5)

    init_analyzers(
        f"http://127.0.0.1:{config['server']['port']}/add_metric", config["metrics"]
    )

    while True:
        schedule.run_pending()
        time.sleep(1)

    # join the server process in case the main process ends
    server_process.join()

if __name__ == "__main__":
    main()