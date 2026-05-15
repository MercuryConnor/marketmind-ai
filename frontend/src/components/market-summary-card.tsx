"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

interface Props {
  companyName: string;
  ticker: string;
  exchange?: string;
  currentPrice: number;
  priceChange: number;
  percentChange: number;
  trend?: number[];
}

export default function MarketSummaryCard(props: Props) {
  const { companyName, currentPrice, priceChange, percentChange } = props;
  const positive = priceChange >= 0;

  return (
    <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm p-6 text-slate-50 transition-all hover:border-slate-700/50 hover:bg-slate-800/50">
      <div className="flex flex-col gap-4">
        {/* Top row */}
        <div className="flex items-center justify-between">
          <span className="font-bold text-white">{companyName}</span>
          <Button
            variant="outline"
            size="sm"
            className="h-8 text-xs border-slate-700 text-slate-400 hover:text-white hover:bg-slate-800"
          >
            View More
          </Button>
        </div>

        {/* Middle row */}
        <div className="flex items-baseline gap-3">
          <span className="text-4xl font-bold text-white">
            ${currentPrice.toFixed(2)}
          </span>
          <span className="text-emerald-500 font-medium text-lg">
            {percentChange.toFixed(2)}% ↗
          </span>
        </div>

        {/* Bottom row */}
        <div className="text-sm text-emerald-500/70">
          {positive ? "+" : "-"}${Math.abs(priceChange).toFixed(2)} (This Week)
        </div>
      </div>
    </Card>
  );
}
