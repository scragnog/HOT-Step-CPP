// Sidebar.tsx — Navigation sidebar
//
// Shows: logo, nav links, connection status, model info, quit button.

import React, { useEffect, useState } from 'react';
import { modelApi, shutdownApi } from '../../services/api';
import type { AceModels } from '../../types';
import './Sidebar.css';

interface SidebarProps {
  activeView: string;
  onViewChange: (view: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onViewChange }) => {
  const [models, setModels] = useState<AceModels | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const check = async () => {
      try {
        const m = await modelApi.list();
        setModels(m);
        setConnected(true);
      } catch {
        setConnected(false);
      }
    };
    check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);

  const handleQuit = async () => {
    if (!confirm('Shut down HOT-Step and ace-server?')) return;
    try {
      await shutdownApi.quit();
    } catch {
      // Server is shutting down, connection will drop — that's expected
    }
  };

  const navItems = [
    { id: 'create', label: '✨ Create', icon: '🎵' },
    { id: 'library', label: '📚 Library', icon: '📚' },
  ];

  return (
    <aside className="app-sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">⚡</div>
        <div>
          <div className="sidebar-logo-title">HOT-Step</div>
          <div className="sidebar-logo-subtitle">CPP Engine</div>
        </div>
      </div>

      <div className="divider" />

      {/* Navigation */}
      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            className={`sidebar-nav-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => onViewChange(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div style={{ flex: 1 }} />

      {/* Connection status */}
      <div className="sidebar-status">
        <div className={`sidebar-status-dot ${connected ? 'connected' : ''}`} />
        <span>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {/* Model info */}
      {models && (
        <div className="sidebar-models">
          <div className="sidebar-models-label">Models</div>
          <div className="sidebar-models-count">
            {models.models.dit.length} DiT · {models.models.lm.length} LM
          </div>
          {models.adapters.length > 0 && (
            <div className="sidebar-models-count">
              {models.adapters.length} Adapter{models.adapters.length !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      )}

      {/* Quit button */}
      <div className="sidebar-quit">
        <button className="btn btn-danger btn-sm w-full" onClick={handleQuit}>
          ⏻ Quit
        </button>
      </div>
    </aside>
  );
};

