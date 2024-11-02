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
            detection_names = data.get("detection_names")
            if not group or not isinstance(group, str):
                raise ValueError("Invalid or missing 'group' parameter.")
            
            
            if detection_names is None:
                self.logger.info(
                    f"Deleting group: {group}"
                )
                self.manager.delete_group(group)
            else:
                self.logger.info(
                    f"Deleting metric: {group}, {detection_names}"
                )
                self.manager.delete_metric(group, detection_names)

            self.write(
                {
                    "status": "success",
                    "message": f"Delete metric {group} {detection_names} successfully",
                }
            )

        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"status": "error", "message": "Invalid JSON"})
        except Exception as e:
            self.logger.error(f"Error adding new metric: {str(e)}")
            self.write({"status": "error", "message": str(e)})


