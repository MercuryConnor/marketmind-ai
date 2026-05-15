"use client";

import React, { useId } from "react";

interface SentimentGaugeProps {
  score: number;
}

function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = (angleDeg * Math.PI) / 180;
  return {
    x: cx + r * Math.cos(rad),
    y: cy + r * Math.sin(rad),
  };
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, startAngle);
  const end = polarToCartesian(cx, cy, r, endAngle);
  const largeArcFlag = Math.abs(endAngle - startAngle) > 180 ? 1 : 0;
  const sweepFlag = endAngle < startAngle ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} ${sweepFlag} ${end.x} ${end.y}`;
}

export default function SentimentGauge({ score }: SentimentGaugeProps) {
  const clamped = Math.max(0, Math.min(100, score));
  const gradientId = useId();

  const cx = 120;
  const cy = 120;
  const radius = 82;

  // Top half: 180deg (left) to 0deg (right)
  const startAngle = 180;
  const endAngle = 0;
  const scoreAngle = 180 - (clamped / 100) * 180;

  const backgroundArc = describeArc(cx, cy, radius, startAngle, endAngle);
  const scoreArc = describeArc(cx, cy, radius, startAngle, scoreAngle);

  const needlePoint = polarToCartesian(cx, cy, radius - 12, scoreAngle);

  return (
    <div className="w-full flex flex-col items-center">
      <svg viewBox="0 0 240 150" className="w-full max-w-[260px] h-auto" role="img" aria-label={`Sentiment score ${clamped} out of 100`}>
        <defs>
          <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="45%" stopColor="#eab308" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>

        <path
          d={backgroundArc}
          fill="none"
          stroke="#1e293b"
          strokeWidth="16"
          strokeLinecap="round"
          transform="translate(0,-18)"
        />

        <path
          d={scoreArc}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth="16"
          strokeLinecap="round"
          transform="translate(0,-18)"
        />

        <line
          x1={cx}
          y1={cy - 18}
          x2={needlePoint.x}
          y2={needlePoint.y - 18}
          stroke="#e2e8f0"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy - 18} r="5" fill="#10b981" />
      </svg>

      <div className="mt-2 text-emerald-500 font-bold">Bullish</div>
      <div className="text-xs text-slate-400">{clamped}/100</div>
    </div>
  );
}
