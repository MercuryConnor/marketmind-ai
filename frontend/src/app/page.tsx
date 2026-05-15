"use client";

import { useState } from "react";
import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send, Zap, ChevronRight, ArrowRight } from "lucide-react";

// Adjust these imports based on where you moved the components
import PriceTrendChart from "@/components/price-trend-chart";
import MarketSummaryCard from "@/components/market-summary-card";
import KeyMetricsCard from "@/components/key-metrics-card";
import SentimentGauge from "@/components/sentiment-gauge";

const mockHistoricalData = [
  { date: "May 17", price: 860, volume: 45000000 },
  { date: "May 18", price: 885, volume: 48000000 },
  { date: "May 19", price: 875, volume: 42000000 },
  { date: "May 20", price: 890, volume: 51000000 },
  { date: "May 21", price: 920, volume: 60000000 },
  { date: "May 22", price: 915, volume: 58000000 },
  { date: "May 23", price: 924.79, volume: 65000000 },
];

const mockMetrics = {
  price: "$924.79",
  marketCap: "$2.24T",
  volume: "52.34M",
  high52W: "$974.00",
  low52W: "$439.20",
  peRatio: "66.21",
  eps: "1.80",
  dividendYield: "0.02",
};

export default function MarketDashboard() {
  const [ticker, setTicker] = useState("NVDA");
  const [query, setQuery] = useState("How did NVIDIA perform this week and what are the key factors behind the movement?");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 p-4 md:p-6 font-sans selection:bg-emerald-500/30">
      {/* Top Navigation */}
      <header className="flex items-center justify-between mb-8 pb-4 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <Image src="/logo.png" width={36} height={36} alt="MarketMind Logo" className="rounded-md" />
          <div className="flex flex-col">
            <span className="font-bold text-xl tracking-tight text-slate-50 leading-tight">MarketMind</span>
            <span className="text-[11px] text-emerald-500/70 font-medium uppercase tracking-wider">Financial AI Assistant</span>
          </div>
        </div>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2 bg-slate-900/50 px-3 py-1.5 rounded-lg border border-slate-800/60">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]"></span>
            <span className="text-slate-300">Live Market: <strong className="text-white">ON</strong></span>
          </div>
          <Button variant="outline" size="sm" className="border-slate-800/60 bg-slate-900/50 hover:bg-slate-800 text-slate-300 rounded-lg">Theme</Button>
        </div>
      </header>

      {/* Main Layout Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-[1600px] mx-auto">
        
        {/* LEFT COLUMN */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          
          {/* Query Bar Layout */}
          <div className="flex flex-col md:flex-row gap-4 w-full">
            {/* Item 1: Stock Input */}
            <div className="flex-shrink-0 bg-slate-900/40 border border-slate-800/60 rounded-2xl p-2 flex items-center gap-3 backdrop-blur-sm">
              <span className="text-sm text-slate-300 font-bold pl-2">Stock</span>
              <div className="flex items-center bg-emerald-500 rounded-xl px-1">
                <input
                  id="ticker-input"
                  type="text"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  className="w-14 bg-transparent text-slate-950 font-black focus:outline-none uppercase text-sm text-center"
                  maxLength={5}
                />
                <Button size="icon" variant="ghost" className="w-6 h-6 text-slate-950 hover:bg-transparent hover:text-slate-800 p-0">
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Item 2: Query Input */}
            <div className="flex-1 bg-slate-900/40 border border-slate-800/60 rounded-2xl flex items-center pr-2 backdrop-blur-sm focus-within:border-emerald-500/50 transition-colors">
              <Input 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full border-0 bg-transparent shadow-none focus-visible:ring-0 text-sm py-6 pl-4 text-slate-200 placeholder:text-slate-500"
                placeholder="How did NVIDIA perform this week and what are the key factors behind the movement?"
              />
              <Button size="icon" variant="ghost" className="text-emerald-500 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors rounded-xl">
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
          {/* End Query Bar Layout */}

          <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-emerald-500 flex items-center gap-2 text-lg">
                <Zap className="w-5 h-5 fill-emerald-500/20" /> AI Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <p className="text-sm leading-relaxed text-slate-200">
                NVIDIA (NVDA) had a strong week, closing at $924.79, up <span className="text-emerald-500 font-medium">6.21%</span> from last week. The stock showed upward momentum driven by strong demand for AI chips...
              </p>
              <div>
                <h4 className="text-emerald-500 font-medium mb-3 text-sm">Key Takeaways</h4>
                <ul className="space-y-2 text-slate-300 list-disc list-inside text-sm">
                  <li>Strong AI and data center demand continues to drive growth</li>
                  <li>Positive analyst sentiment and upward price revisions</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-auto">
             <PriceTrendChart data={mockHistoricalData} />
             <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm">
                <CardHeader className="py-3 px-4 border-b border-slate-800/50">
                  <CardTitle className="text-emerald-500 text-sm flex items-center justify-between">
                    <span>Recent Stock News</span>
                    <span className="text-slate-500 text-xs hover:text-white cursor-pointer">View All</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 p-4 overflow-y-auto">
                  <ul className="space-y-4">
                    <li className="space-y-1">
                      <h5 className="text-sm text-slate-200 font-medium">NVIDIA announces new chip architecture</h5>
                      <p className="text-xs text-slate-500">Reuters • 2 hours ago</p>
                    </li>
                    <li className="space-y-1">
                      <h5 className="text-sm text-slate-200 font-medium">Tech stocks rally led by semiconductor surge</h5>
                      <p className="text-xs text-slate-500">Bloomberg • 5 hours ago</p>
                    </li>
                    <li className="space-y-1">
                      <h5 className="text-sm text-slate-200 font-medium">Analysts raise price targets for major AI hardware providers</h5>
                      <p className="text-xs text-slate-500">CNBC • 1 day ago</p>
                    </li>
                  </ul>
                </CardContent>
             </Card>
          </div>
        </div>

        <div className="lg:col-span-4 flex flex-col gap-6">
          <MarketSummaryCard 
            companyName="NVIDIA Corporation" 
            ticker="NVDA" 
            exchange="NASDAQ" 
            currentPrice={924.79} 
            priceChange={54.19} 
            percentChange={6.21} 
          />
          <KeyMetricsCard metrics={mockMetrics} />
          <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm">
            <CardHeader className="py-4">
              <CardTitle className="text-emerald-500 text-sm">AI Sentiment Insight</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-col items-center justify-center py-4">
                <SentimentGauge score={85} />
              </div>
              
              <div className="space-y-2 pt-2 border-t border-slate-800/50">
                <h4 className="text-sm font-medium text-slate-200">NVDA momentum outlook</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Momentum remains constructive with buyers aggressively defending key support zones. 
                  Technical indicators suggest a continuation of the primary trend as volume profiles 
                  support current price discovery levels.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-2 pt-2">
                <div className="space-y-1">
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Support</p>
                  <p className="text-sm font-semibold text-slate-200">$880</p>
                </div>
                <div className="space-y-1">
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Resistance</p>
                  <p className="text-sm font-semibold text-slate-200">$950</p>
                </div>
                <div className="space-y-1">
                  <p className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Outlook</p>
                  <p className="text-sm font-bold text-emerald-500">Strong Buy</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
