import tornado.web

from engine.server.handler.add_metric_handler import AddMetricHandler
from engine.server.handler.get_predicted_data_handler import GetPredictedDataHandler
from engine.server.handler.delete_metric_handler import DeleteMetricHandler
from engine.server.handler.get_all_metric_promql import GetDetectionJobHandler
from engine.server.handler.get_all_group_handler import GetAllGroupsHandler

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
                r"/get_metrics",
                GetDetectionJobHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
            (
                r"/groups",
                GetAllGroupsHandler,
                dict(logger=logger, analyzer_manager=manager),
            ),
        ],
        settings={},
    )
