/* ─── ImageCompare — drag to reveal the severity mask ──────────
   The original photo and the OpenCV overlay share an aspect ratio
   (the overlay is the same frame, resized), so they can be stacked
   and the top one clipped. A native range input drives the split,
   which gives keyboard and screen-reader support for free.
   ──────────────────────────────────────────────────────────── */

import { useState } from 'react';
import './ImageCompare.css';

export default function ImageCompare({
  original,
  overlay,
  maxHeight = 440,
}: {
  original: string;
  overlay: string | null;
  maxHeight?: number;
}) {
  const [split, setSplit] = useState(50);
  const [ratio, setRatio] = useState(4 / 3);

  const frameStyle = {
    aspectRatio: String(ratio),
    width: `min(100%, ${Math.round(maxHeight * ratio)}px)`,
  };

  if (!overlay) {
    return (
      <div className="compare" style={frameStyle}>
        <img
          src={original}
          alt="Uploaded leaf"
          onLoad={e => setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight || 4 / 3)}
        />
      </div>
    );
  }

  return (
    <div className="compare" style={frameStyle}>
      <img
        src={original}
        alt="Uploaded leaf"
        onLoad={e => setRatio(e.currentTarget.naturalWidth / e.currentTarget.naturalHeight || 4 / 3)}
      />
      <img
        src={overlay}
        alt="Severity mask: green healthy, yellow chlorotic, red necrotic"
        className="compare-top"
        style={{ clipPath: `inset(0 0 0 ${split}%)` }}
      />
      <div className="compare-handle" style={{ left: `${split}%` }} aria-hidden="true">
        <span className="compare-knob">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="m9 18-6-6 6-6M15 6l6 6-6 6" />
          </svg>
        </span>
      </div>
      <span className="compare-tag compare-tag-left">Original</span>
      <span className="compare-tag compare-tag-right">Severity mask</span>
      <input
        type="range"
        min={0}
        max={100}
        step={1}
        value={split}
        onChange={e => setSplit(Number(e.target.value))}
        className="compare-range"
        aria-label="Drag to compare the original photo with the severity mask"
      />
    </div>
  );
}
