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
            group = data.get("group")
            detection_name = data.get("detection_name")
            if not group or not isinstance(group, str):
                raise ValueError("Invalid or missing 'group' parameter.")
            
            
            if detection_name is None:
                self.logger.info(
                    f"Deleting group: {group}"
                )
                self.manager.delete_group(group)
            else:
                self.logger.info(
                    f"Deleting metric: {group}, {detection_name}"
                )
                self.manager.delete_metric(group, detection_name)

            self.write(
                {
                    "status": "success",
                    "message": f"Delete metric {group} {detection_name} successfully",
                }
            )

        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"status": "error", "message": "Invalid JSON"})
        except Exception as e:
            self.logger.error(f"Error adding new metric: {str(e)}")
            self.write({"status": "error", "message": str(e)})


