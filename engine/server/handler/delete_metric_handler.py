import json
import re
import tornado.web
from engine.manager import AnalyzeManager


class DeleteMetricHandler(tornado.web.RequestHandler):
    """Handler to delete metrics."""

    def initialize(self, logger, analyzer_manager: AnalyzeManager):
        """Check if new predicted values are available in the queue before the get request."""
        self.logger = logger
        self.manager = analyzer_manager

    async def post(self):
        """Delete a metric."""
        try:
            # Parse JSON body
            data = json.loads(self.request.body)
            metric_promql = data.get("metric")
            if not metric_promql or not isinstance(metric_promql, str):
                raise ValueError("Invalid or missing 'metric' parameter.")

            self.logger.info(
                f"Deleting metric: {metric_promql}"
            )

            self.manager.delete_metric(metric_promql)

            self.write(
                {
                    "status": "success",
                    "message": f"Delete metric {metric_promql} successfully",
                }
            )

        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"status": "error", "message": "Invalid JSON"})
        except Exception as e:
            self.logger.error(f"Error adding new metric: {str(e)}")
            self.write({"status": "error", "message": str(e)})


