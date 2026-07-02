# چغندر آفت‌زده — Beet Disease Classifier with Grad-CAM

تشخیص بیماری چغندر با استفاده از ResNet-50 Transfer Learning و Grad-CAM برای نمایش نواحی مهم تصویر.

## ساختار پروژه

```
grad_cam/
├── .github/workflows/ci.yml   # CI/CD خودکار
├── backend/                   # Django + TensorFlow
│   ├── config/                # Django settings
│   ├── predictor/             # Main app (views, model, Grad-CAM)
│   │   ├── views.py
│   │   ├── model.py
│   │   ├── gradcam.py
│   │   └── urls.py
│   ├── tests/                 # pytest tests
│   ├── conftest.py            # Test fixtures
│   └── requirements.txt
├── website/                   # React + Vite + Tailwind
│   ├── src/
│   │   ├── components/
│   │   └── __tests__/         # Vitest tests
│   └── package.json
├── grad_cam_app.py            # Standalone Gradio app
├── make_model.py              # Model training script
├── Dockerfile
├── docker-compose.yml
└── best_resnet50_beet_model-99.keras  # Tracked via Git LFS
``>

## راهنمای اجرا

### بدون Docker

```bash
# 1. Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Frontend
cd website
npm install
npm run build
cp dist/index.html ../backend/templates/

# 3. اجرا
cd backend
python manage.py runserver
# باز کردن http://localhost:8000
```

### با Docker

```bash
docker compose up backend
# باز کردن http://localhost:8000
```

### اپلیکیشن Gradio (مستقل)

```bash
python grad_cam_app.py
# باز کردن http://localhost:7860
```

## تست‌ها

### Backend (pytest)

```bash
cd backend
pip install -r requirements-test.txt
pytest tests/ -v
# با coverage:
pytest tests/ -v --cov=predictor --cov-report=term-missing
```

### Frontend (Vitest)

```bash
cd website
npm test                    # یکبار اجرا
npm run test:watch          # حالت watch
npm run test:coverage       # با coverage
```

### همه تست‌ها با Docker

```bash
docker compose run test-backend
```

## CI/CD

در هر Push یا Pull Request به `main`/`master`:

- **Backend Tests**: pytest روی Python 3.10/3.11/3.12
- **Frontend Tests**: Vitest روی Node.js 20/22
- **Lint**: ruff برای کد Python
- **Type Check**: TypeScript برای کد React

## Git LFS

فایل‌های مدل بزرگ (`.keras`, `.h5`, `.tflite`, `.onnx`) از طریق Git LFS ردیابی می‌شوند.

```bash
# نصب Git LFS (یکبار)
git lfs install
```

## متغیرهای محیطی

| متغیر | پیش‌فرض | توضیح |
|--------|---------|-------|
| `TF_CPP_MIN_LOG_LEVEL` | `3` | سطح لاگ TensorFlow |
| `DEBUG` | `True` | حالت debug Django |
