import { HashRouter, Routes, Route } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import PredictPage from './components/PredictPage';

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/predict" element={<PredictPage />} />
      </Routes>
    </HashRouter>
  );
}