/* ─── Navbar — top bar on desktop, bottom tab bar on phones ── */

import { NavLink } from 'react-router-dom';
import { useBackend, useTheme } from '../hooks';
import Icon, { type IconName } from './Icon';
import './Navbar.css';

const LINKS: { to: string; label: string; icon: IconName; end?: boolean }[] = [
  { to: '/', label: 'Home', icon: 'home', end: true },
  { to: '/diagnose', label: 'Diagnose', icon: 'camera' },
  { to: '/history', label: 'History', icon: 'history' },
];

function BackendStatus() {
  const backend = useBackend();
  const { label, tone, title } =
    backend.status === 'checking'
      ? { label: 'Connecting', tone: 'checking', title: 'Checking the analysis server…' }
      : backend.status === 'offline'
        ? { label: 'Offline', tone: 'offline', title: 'Analysis server unreachable — click to retry' }
        : backend.readiness.components.degraded.length > 0
          ? {
              label: 'Limited',
              tone: 'limited',
              title: `Running with fallbacks:\n• ${backend.readiness.components.degraded.join('\n• ')}`,
            }
          : { label: 'Online', tone: 'online', title: 'All components ready' };

  return (
    <button
      type="button"
      className={`status-chip status-${tone}`}
      title={title}
      onClick={backend.status === 'offline' ? backend.retry : undefined}
      aria-label={`Server status: ${label}. ${title}`}
    >
      <span className="status-chip-dot" />
      <span className="status-chip-label">{label}</span>
    </button>
  );
}

export default function Navbar() {
  const { effective, toggle } = useTheme();

  return (
    <>
      <header className="navbar no-print">
        <div className="navbar-inner container">
          <NavLink to="/" className="navbar-brand" aria-label="AgroVision home">
            <img src="/leaf.svg" alt="" width="30" height="30" />
            <span>AgroVision</span>
          </NavLink>

          <nav className="navbar-links" aria-label="Main">
            {LINKS.map(link => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <Icon name={link.icon} size={17} />
                <span>{link.label}</span>
              </NavLink>
            ))}
          </nav>

          <div className="navbar-tools">
            <BackendStatus />
            <button
              type="button"
              className="btn btn-ghost btn-icon"
              onClick={toggle}
              aria-label={`Switch to ${effective === 'dark' ? 'light' : 'dark'} theme`}
              title={`Switch to ${effective === 'dark' ? 'light' : 'dark'} theme`}
            >
              <Icon name={effective === 'dark' ? 'sun' : 'moon'} size={18} />
            </button>
          </div>
        </div>
      </header>

      <nav className="tabbar no-print" aria-label="Main">
        {LINKS.map(link => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.end}
            className={({ isActive }) => `tab-link ${isActive ? 'active' : ''}`}
          >
            <Icon name={link.icon} size={22} />
            <span>{link.label}</span>
          </NavLink>
        ))}
      </nav>
    </>
  );
}
