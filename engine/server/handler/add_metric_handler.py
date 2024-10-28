import json
import re
import tornado.web
from engine.manager import AnalyzeManager
from configuration import Configuration


class AddMetricHandler(tornado.web.RequestHandler):
    """Handler to add new metrics for training."""

    def initialize(self, logger, analyzer_manager: AnalyzeManager):
        """Check if new predicted values are available in the queue before the get request."""
        self.logger = logger
        self.manager = analyzer_manager

    async def post(self):
        """Add a new metric and trigger model training."""
        try:
            # Parse JSON body
            data = json.loads(self.request.body)
            metric_promql, model_name, window_size = validate_parameters(data)

            self.logger.info(
                f"Received new metric for training: {metric_promql}, model: {model_name}"
            )

            self.manager.add_metric(
                metric_promql,
                model_name,
                Configuration.prometheus_url,
                window_size,
                Configuration.retraining_interval_minutes,
                300,
            )

            self.write(
                {
                    "status": "success",
                    "message": f"Add metric {metric_promql} successfully",
                }
            )

        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"status": "error", "message": "Invalid JSON"})
        except Exception as e:
            self.logger.error(f"Error adding new metric: {str(e)}")
            self.write({"status": "error", "message": str(e)})


def validate_parameters(data):
    """Validate the input parameters."""
    # Validate new_metric
    new_metric = data.get("metric")
    if not new_metric or not isinstance(new_metric, str):
        raise ValueError("Invalid or missing 'metric' parameter.")

    # Validate model_name
    model_name = data.get("model")
    valid_models = {"prophet", "fourier", "lstm", "sarima"}
    if not model_name or model_name not in valid_models:
        raise ValueError(
            f"Invalid or missing 'model' parameter. Must be one of {valid_models}."
        )

    # Validate window_size
    window_size = data.get("window_size")
    if (
        not window_size
        or not isinstance(window_size, str)
        or not re.match(r"^\d+[dhm]$", window_size)
    ):
        raise ValueError(
            "Invalid or missing 'window_size' parameter. Must be a string like '10d', '5h', or '30m'."
        )

    return new_metric, model_name, window_size
