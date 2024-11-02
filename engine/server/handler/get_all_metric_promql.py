import tornado
from engine.manager import AnalyzeManager
from prometheus_client import generate_latest, REGISTRY


class GetDetectionJobHandler(tornado.web.RequestHandler):
    """Tornado web request handler."""

    def initialize(self, logger, analyzer_manager: AnalyzeManager):
        """Check if new predicted values are available in the queue before the get request."""
        self.logger = logger
        self.manager = analyzer_manager

    async def get(self):
        group_name = self.get_argument("group_name", None)
        
        if not group_name:
            self.set_header("Content-Type", "text; charset=utf-8")
            self.write(
                {
                    "status": "success",
                    "message": f"Get all metric promql for all groups successfully",
                    "data": self.manager.get_all_metric_promql(),
                }
            )
        else:
            self.set_header("Content-Type", "text; charset=utf-8")
            self.write(
                {
                    "status": "success",
                    "message": f"Get all metric promql for group {group_name} successfully",
                    "data": self.manager.get_detection_group(group_name),
                }
            )
