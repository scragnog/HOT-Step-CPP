# Waveform Player — wavesurfer.js Integration

## Overview
Replace the simple seekbar in the bottom Player with a full-width wavesurfer.js waveform visualizer in Bars mode with Hover plugin. Section markers from LRC files displayed above the waveform.

## Layout
```
┌──────────────────────────────────────────┐
│  Sidebar │ Content Area                  │
├──────────┴───────────────────────────────┤
│  Section Markers (h-5, "Intro|Verse 1…") │
│  Waveform (wavesurfer bars, ~h-16)       │
│  Transport Controls (song info|btns|vol) │
└──────────────────────────────────────────┘
```

## Technology
- wavesurfer.js v7 (Bars mode, Hover plugin, Regions plugin for markers)
- Wavesurfer owns the `<audio>` element

## Components
- `WaveformPlayer.tsx` — wavesurfer wrapper
- `SectionMarkers.tsx` — LRC-based section labels (ported from Python app)
- `Player.tsx` — transport controls only (seekbar removed)
- `App.tsx` — restructured bottom area, audio management via wavesurfer

## Colours
- Unplayed: `rgba(113, 113, 122, 0.5)`
- Played: pink-500 `#ec4899` → purple-500 `#a855f7` gradient
- Cursor: `#ec4899`
- Hover: `#a855f7`
