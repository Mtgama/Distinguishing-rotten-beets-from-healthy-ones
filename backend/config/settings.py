"""
Django settings for the beet classifier backend.

The React frontend is built to a single self-contained index.html
(vite-plugin-singlefile) and served from the templates dir, so the whole
app runs under one Django server on port 8000 -- no separate dev server,
no CORS configuration needed.
"""

from pathlib import Path

# backend/
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "django-insecure-change-me-this-is-a-local-dev-project"

DEBUG = True

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "*"]


# ---- Applications ----
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "predictor",
]

MIDDLEWARE = [
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# ---- Database ----
# Not needed for this app, but Django requires the setting.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# ---- Static files ----
STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Raise the default upload limit so large plant photos are accepted.
# (Not strictly enforced in DEBUG without a data validation step, but kept
# for clarity.)
DATA_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024
