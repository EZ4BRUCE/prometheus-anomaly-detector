import json
import re
import tornado.web
from engine.manager import AnalyzeManager


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
            data_list = json.loads(self.request.body)
            for data in data_list:
                (
                    metric_promql,
                    group,
                    detection_name,
                    model_name,
                    window_size,
                    retraining_interval_minutes,
                    sync_new_series_interval_seconds,
                ) = validate_parameters(data)

                future_offset = data.get("future_offset")
                if future_offset is not None:
                    if not isinstance(future_offset, str):
                        raise ValueError(
                            "Invalid or missing 'future_offset' parameter."
                        )
                    self.logger.info(
                        "[%s] Received new metric for training: %s, model: %s with future offset: %s",
                        "server",
                        metric_promql,
                        model_name,
                        future_offset,
                    )
                else:
                    self.logger.info(
                        "[%s] Received new metric for training: %s, model: %s",
                        "server",
                        metric_promql,
                        model_name,
                    )

                # sync_new_series_interval_seconds must be greater than 300 seconds
                if sync_new_series_interval_seconds < 300:
                    sync_new_series_interval_seconds = 300

                self.manager.add_metric(
                    group,
                    detection_name,
                    metric_promql,
                    model_name,
                    self.manager.prometheus_url,
                    future_offset,
                    window_size,
                    retraining_interval_minutes,
                    sync_new_series_interval_seconds,
                )

            self.write(
                {
                    "status": "success",
                    "message": f"Add {len(data_list)} metrics for group{group} successfully",
                }
            )

        except json.JSONDecodeError:
            self.set_status(400)
            self.write({"status": "error", "message": "Invalid JSON"})
        except Exception as e:
            self.logger.error(f"Error adding new metrics: {str(e)}")
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

    # Validate group
    group = data.get("group")
    if not group or not isinstance(group, str):
        raise ValueError("Invalid or missing 'group' parameter.")

    # Validate detection_name
    detection_name = data.get("detection_name")
    if not detection_name or not isinstance(detection_name, str):
        raise ValueError("Invalid or missing 'detection_name' parameter.")

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

    # Validate retraining_interval_minutes
    retraining_interval_minutes = data.get("retraining_interval_minutes")
    if not retraining_interval_minutes or not isinstance(
        retraining_interval_minutes, int
    ):
        raise ValueError("Invalid or missing 'retraining_interval_minutes' parameter.")

    # Validate sync_new_series_interval_seconds
    sync_new_series_interval_seconds = data.get("sync_new_series_interval_seconds")
    if not sync_new_series_interval_seconds or not isinstance(
        sync_new_series_interval_seconds, int
    ):
        raise ValueError(
            "Invalid or missing 'sync_new_series_interval_seconds' parameter."
        )

    return (
        new_metric,
        group,
        detection_name,
        model_name,
        window_size,
        retraining_interval_minutes,
        sync_new_series_interval_seconds,
    )
