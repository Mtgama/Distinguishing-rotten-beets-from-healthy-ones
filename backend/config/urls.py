"""Root URL configuration.

Routes:
    /api/predict/   -> predictor.predict (JSON, multipart image upload)
    /               -> predictor.index   (serves the built React index.html)
"""

from django.urls import include, path

from predictor.views import index

urlpatterns = [
    path("api/", include("predictor.urls")),
    # SPA root: any non-API path renders the React app.
    path("", index, name="index"),
]
