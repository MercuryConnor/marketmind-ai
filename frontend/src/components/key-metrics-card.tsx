"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { CheckCircle2 } from "lucide-react";

interface Metrics {
  price: number | string;
  marketCap: number | string;
  volume: number | string;
  high52W: number | string;
  low52W: number | string;
  peRatio: number | string;
  eps: number | string;
  dividendYield: number | string;
}

interface Props {
  metrics: Metrics;
}

function formatNumber(v: number | string) {
  if (typeof v === "number") {
    if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
    if (v >= 1e9) return `$${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    return v.toLocaleString();
  }
  return v;
}

function formatMoney(v: number | string) {
  if (typeof v === "number") return `$${v.toFixed(2)}`;
  return v;
}

export default function KeyMetricsCard({ metrics }: Props) {
  const rows: [string, React.ReactNode][] = [
    ["Price", formatMoney(metrics.price)],
    ["Market Cap", formatNumber(metrics.marketCap)],
    ["Volume", formatNumber(metrics.volume)],
    ["52W High", formatMoney(metrics.high52W)],
    ["52W Low", formatMoney(metrics.low52W)],
    ["P/E Ratio", typeof metrics.peRatio === "number" ? metrics.peRatio.toFixed(2) : metrics.peRatio],
    ["EPS (TTM)", typeof metrics.eps === "number" ? metrics.eps.toFixed(2) : metrics.eps],
    ["Dividend Yield", typeof metrics.dividendYield === "number" ? `${metrics.dividendYield.toFixed(2)}%` : metrics.dividendYield],
  ];

  return (
    <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm text-slate-50 transition-all hover:border-slate-700/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-slate-100">Key Metrics</CardTitle>
      </CardHeader>

      <CardContent>
        <div className="space-y-1">
          {rows.map(([label, value], index) => (
            <div
              key={label}
              className={`flex justify-between items-center py-2.5 ${index !== rows.length - 1 ? "border-b border-slate-800/50" : ""}`}
            >
              <div className="text-xs text-slate-400">{label}</div>
              <div className="text-xs text-slate-200 font-medium">{value}</div>
            </div>
          ))}
        </div>

        <div className="flex items-center gap-1.5 mt-4 pt-4 border-t border-slate-800/50">
          <CheckCircle2 className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs text-slate-500">Data provided by Finnhub</span>
        </div>
      </CardContent>
    </Card>
  );
}
