"use client";

import React, { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

type Timeframe = "1D" | "1W" | "1M" | "1Y";

export interface HistoricalPoint {
  date: string | number | Date;
  close?: number;
  price?: number;
  volume?: number;
}

interface Props {
  data: HistoricalPoint[];
  height?: number;
}

const TIMEFRAMES: Timeframe[] = ["1D", "1W", "1M", "1Y"];

function toTimestamp(d: string | number | Date) {
  return typeof d === "number" ? d : new Date(d).getTime();
}

function toValue(p: HistoricalPoint) {
  if (typeof p.close === "number") return p.close;
  if (typeof p.price === "number") return p.price;
  return 0;
}

function formatXAxisLabel(ts: number, timeframe: Timeframe) {
  const d = new Date(ts);
  if (timeframe === "1D") {
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  if (timeframe === "1W" || timeframe === "1M") {
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }
  return d.getFullYear().toString();
}

function formatTooltipLabel(ts: number) {
  const d = new Date(ts);
  return d.toLocaleString();
}

export default function PriceTrendChart({ data, height = 300 }: Props) {
  const [timeframe, setTimeframe] = useState<Timeframe>("1M");

  const normalized = useMemo(() => {
    return data
      .map((p) => ({ _ts: toTimestamp(p.date), value: toValue(p), volume: p.volume }))
      .sort((a, b) => a._ts - b._ts);
  }, [data]);

  const filtered = useMemo(() => {
    if (!normalized.length) return [];

    const latestTimestamp = normalized[normalized.length - 1]?._ts ?? 0;
    let start = latestTimestamp;
    switch (timeframe) {
      case "1D":
        start = latestTimestamp - 1000 * 60 * 60 * 24;
        break;
      case "1W":
        start = latestTimestamp - 1000 * 60 * 60 * 24 * 7;
        break;
      case "1M":
        start = latestTimestamp - 1000 * 60 * 60 * 24 * 30;
        break;
      case "1Y":
        start = latestTimestamp - 1000 * 60 * 60 * 24 * 365;
        break;
    }

    const out = normalized
      .filter((p) => p._ts >= start)
      .map((p) => ({ date: p._ts, value: p.value, volume: p.volume }));

    if (out.length === 0) {
      const last = normalized.slice(-Math.min(30, normalized.length));
      return last.map((p) => ({ date: p._ts, value: p.value, volume: p.volume }));
    }

    return out;
  }, [normalized, timeframe]);

  const accent = "#10B981";

  return (
    <div className="w-full bg-slate-900 rounded-lg p-4 text-slate-200">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium">Price Trend</h3>

        <div className="flex gap-2">
          {TIMEFRAMES.map((tf) => {
            const active = tf === timeframe;
            return (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                aria-pressed={active}
                className={`px-3 py-1.5 text-xs rounded-md font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-emerald-400
                  ${active
                    ? "text-emerald-500 bg-emerald-500/10"
                    : "text-slate-400"}
                `}
              >
                {tf}
              </button>
            );
          })}
        </div>
      </div>

      <div className="w-full h-[250px]" style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={filtered} margin={{ top: 8, right: 12, left: 0, bottom: 6 }}>
            <defs>
              <linearGradient id="priceTrendGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10B981" stopOpacity={0.45} />
                <stop offset="100%" stopColor="#10B981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} stroke="#1e293b" strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              domain={["dataMin", "dataMax"]}
              type="number"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#64748b", fontSize: 12 }}
              tickFormatter={(ts) => formatXAxisLabel(ts as number, timeframe)}
              padding={{ left: 6, right: 6 }}
            />
            <YAxis yAxisId="price" domain={['auto', 'auto']} hide />
            <YAxis yAxisId="volume" hide />
            <Tooltip
              contentStyle={{ backgroundColor: "#0f172a", borderRadius: 6, border: "1px solid rgba(255,255,255,0.06)" }}
              itemStyle={{ color: accent }}
              labelStyle={{ color: "#e2e8f0" }}
              formatter={(value, name) => {
                const numeric = typeof value === "number" ? value : Number(value);
                if (name === "volume") return [numeric.toLocaleString(), "Volume"];
                return [`$${Number.isFinite(numeric) ? numeric.toFixed(2) : "0.00"}`, "Price"];
              }}
              labelFormatter={(label) => {
                const numeric = typeof label === "number" ? label : Number(label);
                return Number.isFinite(numeric) ? formatTooltipLabel(numeric) : "";
              }}
            />
            <Area
              type="monotone"
              dataKey="value"
              yAxisId="price"
              stroke="#10B981"
              strokeWidth={2}
              fill="url(#priceTrendGradient)"
              dot={false}
              activeDot={{ r: 5, stroke: "#fff", strokeWidth: 2, fill: "#10B981" }}
            />
            <Bar dataKey="volume" yAxisId="volume" fill="#10b981" opacity={0.3} maxBarSize={8} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
