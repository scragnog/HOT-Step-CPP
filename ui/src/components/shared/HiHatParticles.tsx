// HiHatParticles.tsx — Tiny sparkle particles that float upward on hi-hat energy.
// Positioned over the player/waveform area. Uses CSS @keyframes for animation,
// JS just spawns/removes DOM elements. Capped at 30 active particles.

import React, { useRef, useEffect, useCallback } from 'react';
import { useDiscoMode } from '@/stores/discoStore';
import { useHihatEnergy } from '@/stores/discoStore';

const MAX_PARTICLES = 30;
const PARTICLE_LIFETIME = 1200; // ms — matches CSS animation duration
const SPAWN_THRESHOLD = 0.08;   // Minimum energy to spawn particles

export const HiHatParticles: React.FC = () => {
  const discoMode = useDiscoMode();
  const hihatEnergy = useHihatEnergy();
  const containerRef = useRef<HTMLDivElement>(null);
  const activeCount = useRef(0);

  const spawnParticle = useCallback(() => {
    const container = containerRef.current;
    if (!container || activeCount.current >= MAX_PARTICLES) return;

    const el = document.createElement('div');
    el.className = 'disco-particle';

    // Random position across container width
    const x = Math.random() * 100;
    el.style.left = `${x}%`;
    el.style.bottom = `${Math.random() * 20}%`;

    // Random size variation
    const size = 2 + Math.random() * 3;
    el.style.width = `${size}px`;
    el.style.height = `${size}px`;

    // Rainbow colours — full spectrum like the border glow
    const hue = Math.random() * 360;
    el.style.background = `hsl(${hue}, 90%, 70%)`;

    container.appendChild(el);
    activeCount.current++;

    // Remove after animation completes
    setTimeout(() => {
      el.remove();
      activeCount.current--;
    }, PARTICLE_LIFETIME);
  }, []);

  useEffect(() => {
    if (!discoMode || hihatEnergy < SPAWN_THRESHOLD) return;

    // Spawn 1-3 particles proportional to energy
    const count = Math.ceil(hihatEnergy * 3);
    for (let i = 0; i < count; i++) {
      spawnParticle();
    }
  }, [discoMode, hihatEnergy, spawnParticle]);

  if (!discoMode) return null;

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: 10,
      }}
    />
  );
};
