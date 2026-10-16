/* ─── Navbar ───────────────────────────────────────────────── */

import { NavLink } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar" id="main-nav">
      <NavLink to="/" className="navbar-brand">
        <div className="brand-icon">
          <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path
              d="M16 2C9 6 4 14 5 24c0 0 3-2 7-3 0-6 2-12 8-16C17 7 15 12 14 18c3-.5 6-.5 9 0 2-8-1-14-7-16z"
              fill="#22c55e"
              stroke="#166534"
              strokeWidth="1"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <span className="brand-text">AgroVision</span>
      </NavLink>

      <div className="navbar-links">
        <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
            <polyline points="9 22 9 12 15 12 15 22"/>
          </svg>
          <span>Home</span>
        </NavLink>

        <NavLink to="/diagnose" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
          <span>Diagnose</span>
        </NavLink>

        <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <polyline points="12 6 12 12 16 14"/>
          </svg>
          <span>History</span>
        </NavLink>
      </div>
    </nav>
  );
}
