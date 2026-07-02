import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

interface PredictionResult {
  prediction: 'healthy' | 'diseased';
  confidence: number;
  heatmap_url: string;
  visualization_url: string;
}

export default function PredictPage() {
  const navigate = useNavigate();
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>('');

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError('');
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError('');

    try {
      // Django backend endpoint - adjust as needed
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch('/api/predict/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      
      setResult({
        prediction: data.prediction,
        confidence: data.confidence,
        heatmap_url: data.heatmap_url,
        visualization_url: data.visualization_url,
      });
    } catch (err) {
      setError('خطا در پردازش تصویر. لطفاً دوباره تلاش کنید.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const resetState = () => {
    setSelectedImage(null);
    setPreviewUrl('');
    setResult(null);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <button
              onClick={() => navigate('/')}
              className="inline-flex items-center text-slate-600 hover:text-slate-900 mb-6"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              بازگشت به صفحه اصلی
            </button>
            <h1 className="text-3xl font-bold text-slate-900">تشخیص چغندر آفت‌زده</h1>
            <p className="text-slate-500 mt-2">یک تصویر از چغندر خود آپلود کنید</p>
          </div>

          {/* Upload Section */}
          <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
            {!selectedImage ? (
              <div className="border-2 border-dashed border-slate-300 rounded-xl p-12 text-center">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageChange}
                  id="image-upload"
                  className="hidden"
                />
                <label
                  htmlFor="image-upload"
                  className="cursor-pointer flex flex-col items-center"
                >
                  <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mb-4">
                    <svg className="w-8 h-8 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-5.92M15 16a4 4 0 008-4 4 4 0 00-8-4 4 4 0 00-8 4 4 4 0 008 4z" />
                    </svg>
                  </div>
                  <p className="text-slate-600 mb-2">کلیک کنید یا تصویر را بکشید</p>
                  <p className="text-slate-400 text-sm">فرمت: JPG، PNG یا WEBP</p>
                </label>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Image Preview */}
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-slate-900">تصویر انتخاب شده</h3>
                  <button
                    onClick={resetState}
                    className="text-slate-500 hover:text-slate-700"
                  >
                    حذف
                  </button>
                </div>
                <div className="bg-slate-100 rounded-xl p-6 text-center">
                  <img
                    src={previewUrl}
                    alt="Selected"
                    className="max-h-96 mx-auto rounded-lg shadow-lg"
                  />
                </div>

                {/* Predict Button */}
                <button
                  onClick={handlePredict}
                  disabled={isLoading}
                  className="w-full py-4 px-8 text-lg font-semibold text-white bg-gradient-to-r from-emerald-500 to-teal-600 rounded-xl hover:from-emerald-600 hover:to-teal-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={4}></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      در حال پردازش...
                    </>
                  ) : (
                    'پردازش و تشخیص'
                  )}
                </button>

                {error && (
                  <div className="bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">
                    {error}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Results Section */}
          {result && (
            <div className="space-y-8">
              {/* Prediction Result */}
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <div className="text-center mb-6">
                  <div className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 ${
                    result.prediction === 'healthy' 
                      ? 'bg-green-100 text-green-600' 
                      : 'bg-red-100 text-red-600'
                  }`}>
                    {result.prediction === 'healthy' ? (
                      <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    )}
                  </div>
                  <h3 className="text-xl font-semibold text-slate-900">
                    {result.prediction === 'healthy' ? 'چغندر سالم' : 'چغندر آفت‌زده'}
                  </h3>
                  <p className="text-slate-500 mt-2">
                    احتمال تشخیص: <span className="font-bold text-slate-900">{(result.confidence * 100).toFixed(2)}%</span>
                  </p>
                </div>
              </div>

              {/* Visualization Images */}
              <div className="grid md:grid-cols-2 gap-6">
                {/* Grad-CAM Visualization */}
                <div className="bg-white rounded-2xl shadow-xl p-6">
                  <h3 className="font-semibold text-slate-900 mb-4">تصویر تشخیصی (Grad-CAM)</h3>
                  <div className="bg-slate-100 rounded-lg p-4 text-center">
                    <img
                      src={result.visualization_url}
                      alt="Grad-CAM Visualization"
                      className="max-h-64 mx-auto rounded-lg shadow-lg"
                      onError={(e) => {
                        e.currentTarget.src = '';
                        e.currentTarget.className = 'max-h-64 mx-auto flex items-center justify-center text-slate-300';
                        (e.currentTarget as HTMLElement).textContent = 'تصویر تشخیصی';
                      }}
                    />
                  </div>
                  <p className="text-slate-500 text-sm mt-3">
                    تصویر با نمایش نقاط مهم که مدل برای تشخیص استفاده کرده
                  </p>
                </div>

                {/* Heatmap */}
                <div className="bg-white rounded-2xl shadow-xl p-6">
                  <h3 className="font-semibold text-slate-900 mb-4">هیت مپ تصویر</h3>
                  <div className="bg-slate-100 rounded-lg p-4 text-center">
                    <img
                      src={result.heatmap_url}
                      alt="Heatmap"
                      className="max-h-64 mx-auto rounded-lg shadow-lg"
                      onError={(e) => {
                        e.currentTarget.src = '';
                        e.currentTarget.className = 'max-h-64 mx-auto flex items-center justify-center text-slate-300';
                        (e.currentTarget as HTMLElement).textContent = 'هیت مپ';
                      }}
                    />
                  </div>
                  <p className="text-slate-500 text-sm mt-3">
                    نمایش گرمای تشخیصی برای هر بخش از تصویر
                  </p>
                </div>
              </div>

              {/* Confidence Breakdown */}
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <h3 className="font-semibold text-slate-900 mb-6">جزئیات تشخیص</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-slate-600">احتمال چغندر سالم</span>
                      <span className="font-semibold">{((1 - result.confidence) * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full" 
                        style={{ width: `${(1 - result.confidence) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-slate-600">احتمال چغندر آفت‌زده</span>
                      <span className="font-semibold text-red-600">{`${(result.confidence * 100).toFixed(2)}%`}</span>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-red-500 h-2 rounded-full" 
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}