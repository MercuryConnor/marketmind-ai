# Phase 3 UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update right column components (MarketSummaryCard, KeyMetricsCard, Sentiment Insight) with glassmorphic styling and improved layouts.

**Architecture:** We will apply consistent glassmorphic styles across right-column cards, rebuild the MarketSummary layout for better hierarchy, expand the KeyMetrics data points, and enhance the Sentiment card with technical analysis insights.

**Tech Stack:** Next.js (App Router), Tailwind CSS, Lucide Icons.

---

### Task 1: Polish `MarketSummaryCard`

**Files:**
- Modify: `frontend/src/components/market-summary-card.tsx`

- [ ] **Step 1: Rebuild MarketSummaryCard layout**
Update root Card style and completely restructure the header and content area.

```tsx
// frontend/src/components/market-summary-card.tsx

import { Button } from "@/components/ui/button"; // Ensure Button import

export default function MarketSummaryCard(props: Props) {
  const { companyName, ticker, currentPrice, priceChange, percentChange, trend } = props;
  const positive = priceChange >= 0;

  return (
    <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm text-slate-50 transition-all hover:border-slate-700/40">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between mb-4">
          <div className="flex flex-col">
            <span className="text-white font-bold text-base leading-none">{companyName}</span>
            <span className="text-slate-500 text-xs mt-1">{ticker} • NASDAQ</span>
          </div>
          <Button variant="outline" size="sm" className="h-7 px-2 text-[10px] border-slate-800 bg-transparent text-slate-400 hover:text-white hover:bg-slate-800">
            View More
          </Button>
        </div>
        
        <div className="flex items-baseline gap-3">
          <span className="text-4xl font-bold text-white">${currentPrice.toFixed(2)}</span>
          <span className="text-emerald-500 font-medium text-lg flex items-center">
            {percentChange.toFixed(2)}% <span className="ml-0.5 text-xs">↗</span>
          </span>
        </div>
        
        <div className="mt-1 text-sm text-emerald-500/70 font-medium">
          +{priceChange.toFixed(2)} (This Week)
        </div>
      </CardHeader>

      <CardContent>
        <div className="flex items-center justify-between pt-4 border-t border-slate-800/50">
          <div className="w-full h-12 flex items-center">
            <Sparkline values={trend} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

### Task 2: Polish `KeyMetricsCard`

**Files:**
- Modify: `frontend/src/components/key-metrics-card.tsx`

- [ ] **Step 1: Expand metrics and add footer**
Update the root card style, add new metrics to the rows, and append the Finnhub data source footer.

```tsx
// frontend/src/components/key-metrics-card.tsx

import { CheckCircle2 } from "lucide-react"; // Import new icon

export default function KeyMetricsCard({ metrics }: Props) {
  // Update mock rendering/props to include more fields in task 3 as well
  const rows: [string, React.ReactNode][] = [
    ["Price", formatMoney(metrics.price)],
    ["Market Cap", formatNumber(metrics.marketCap)],
    ["Volume", formatNumber(metrics.volume)],
    ["52W High", formatMoney(metrics.high52W)],
    ["52W Low", formatMoney(metrics.low52W)],
    ["P/E Ratio", typeof metrics.peRatio === "number" ? metrics.peRatio.toFixed(2) : metrics.peRatio],
    ["EPS (TTM)", "$12.45"], // Mocked for now
    ["Dividend Yield", "0.02%"], // Mocked for now
  ];

  return (
    <Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm text-slate-50 transition-all">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm text-slate-100">Key Metrics</CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="divide-y divide-slate-800/50">
          {rows.map(([label, value]) => (
            <div key={label} className="flex justify-between items-center py-2.5">
              <div className="text-slate-400 text-xs">{label}</div>
              <div className="text-slate-200 text-xs font-medium">{value}</div>
            </div>
          ))}
        </div>
        
        <div className="pt-2 flex items-center gap-1.5 border-t border-slate-800/50">
          <CheckCircle2 className="w-3 h-3 text-slate-500" />
          <span className="text-[10px] text-slate-500 uppercase tracking-tight">Data provided by Finnhub</span>
        </div>
      </CardContent>
    </Card>
  );
}
```

### Task 3: Polish Sentiment card in `page.tsx`

**Files:**
- Modify: `frontend/src/app/page.tsx`

- [ ] **Step 1: Rebuild Sentiment Insight section**
Rename title, remove news content, add momentum paragraph and support/resistance grid.

```tsx
// frontend/src/app/page.tsx

{/* Replace the Sentiment card in page.tsx */}
<Card className="bg-slate-900/40 border-slate-800/60 rounded-2xl backdrop-blur-sm">
  <CardHeader className="py-4">
    <CardTitle className="text-emerald-500 text-sm">AI Sentiment Insight</CardTitle>
  </CardHeader>
  <CardContent className="space-y-6">
    <div className="flex flex-col items-center justify-center h-32 border border-dashed border-slate-700/30 rounded-xl bg-slate-900/20">
      <SentimentGauge score={85} />
    </div>

    <div className="space-y-4">
      <div className="space-y-1.5">
        <h4 className="text-sm font-medium text-slate-200">NVDA momentum outlook</h4>
        <p className="text-xs text-slate-400 leading-relaxed">
          Momentum remains constructive with buyers aggressively defending key support zones. 
          Technical indicators suggest a continuation of the primary trend as volume profiles 
          support current price discovery levels.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-2 pt-2">
        <div className="space-y-1">
          <span className="block text-[10px] text-slate-500 font-bold uppercase">Support</span>
          <span className="block text-sm text-white font-medium">$880</span>
        </div>
        <div className="space-y-1">
          <span className="block text-[10px] text-slate-500 font-bold uppercase">Resistance</span>
          <span className="block text-sm text-white font-medium">$950</span>
        </div>
        <div className="space-y-1">
          <span className="block text-[10px] text-slate-500 font-bold uppercase">Outlook</span>
          <span className="block text-sm text-emerald-500 font-bold">Strong Buy</span>
        </div>
      </div>
    </div>
  </CardContent>
</Card>
```
