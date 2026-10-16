/* ─── App Shell — routing + layout ─────────────────────────── */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Diagnose from './pages/Diagnose';
import History from './pages/History';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/diagnose" element={<Diagnose />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
