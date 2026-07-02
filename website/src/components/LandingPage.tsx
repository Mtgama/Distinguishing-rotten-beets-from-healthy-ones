import { useNavigate } from 'react-router-dom';

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      {/* Hero Section */}
      <div className="container mx-auto px-4 pt-20 pb-16">
        <div className="text-center max-w-3xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold text-slate-900 mb-6">
            شناسایی چغندر آفت‌زده با هوش مصنوعی
          </h1>
          <p className="text-xl text-slate-600 mb-8">
            مدلی پیشرفته برای تشخیص بیماری‌های چغندر با استفاده از شبکه عصبی کانولوشنی
          </p>
          
          {/* Model Info Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-12">
            <div className="bg-white rounded-2xl p-6 shadow-lg border border-slate-200">
              <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 4.04A12 12 0 003 9c0 5.591 3.824 10.29 9 11.624A12 12 0 0021 12c0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">دقت ۹۹.۵٪</h3>
              <p className="text-slate-500 text-sm">آمار دقت بالای مدل در تشخیص بیماری‌ها</p>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-lg border border-slate-200">
              <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="font-semibold text-slate-900 mb-2">ResNet-50</h3>
              <p className="text-slate-500 text-sm">مبتنی بر معماری شبکه عصبی ResNeXt-50</p>
            </div>

            <div className="bg-white rounded-2xl p-6 shadow-lg border border-slate-200">
              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mb-4 mx-auto">
                <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
<h3 className="font-semibold text-slate-900 mb-2">Grad-CAM</h3>
              <p className="text-slate-500 text-sm">نمایش تحلیل‌شده نقاط مهم تصویر</p>
              <p className="text-slate-500 text-sm">نمایش تحلیل‌شده نقاط مهم تصویر</p>
            </div>
          </div>

          <button
            onClick={() => navigate('/predict')}
            className="inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-white bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all duration-200 shadow-lg shadow-green-200 hover:shadow-xl transform hover:-translate-y-1"
          >
            شروع تست مدل
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5-5 5V7z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-white py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div>
                <h2 className="text-3xl font-bold text-slate-900 mb-6">ویژگی‌های مدل</h2>
                <ul className="space-y-4">
                  <li className="flex items-start">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-slate-600">تشخیص دو نوع چغندر: سالم و آفت‌زده</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-slate-600">نمایش تصویری نقاط مهم با تکنیک Grad-CAM</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-slate-600">محاسبه احتمال تشخیص برای هر کلاس</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-3">✓</span>
                    <span className="text-slate-600">سازگاری کامل با سرور جنگو</span>
                  </li>
                </ul>
              </div>
              <div className="bg-slate-100 rounded-2xl p-8 text-center">
                <img 
                  src="/beet-placeholder.svg" 
                  alt="Sugar Beet" 
                  className="mx-auto w-48 h-48"
                  onError={(e) => {
                    e.currentTarget.src = '';
                    e.currentTarget.className = 'mx-auto w-48 h-48 text-slate-300';
                  }}
                />
                <p className="text-slate-500 mt-4">تصویر نمونه چغندر</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}