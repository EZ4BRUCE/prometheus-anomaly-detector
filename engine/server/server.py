import tornado.web

from engine.server.handler.add_metric_handler import AddMetricHandler
from engine.server.handler.get_predicted_data_handler import GetPredictedDataHandler
from engine.server.handler.delete_metric_handler import DeleteMetricHandler
from engine.server.handler.get_all_metric_promql import GetAllMetricPromqlHandler
def make_app(logger, manager):
    """Initialize the tornado web app."""
    logger.info("Initializing Tornado Web App")
    return tornado.web.Application(
        [
            (
                r"/metrics",
                GetPredictedDataHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
            (
                r"/",
                GetPredictedDataHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
            (
                r"/add_metric",
                AddMetricHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
            (
                r"/delete_metric",
                DeleteMetricHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
            (
                r"/all_metrics",
                GetAllMetricPromqlHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
        ],
        settings={},
    )
