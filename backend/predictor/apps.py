from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    def ready(self):
        # Load the TensorFlow model once when the app starts.
        from . import model

        model.get_model()
