import { ArrowUpRight, BarChart3 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type QuickViewCardProps = {
  data: {
    ticker: string;
    name: string;
    exchange: string;
    price: string;
    weeklyChange: string;
  };
};

export function QuickViewCard({ data }: QuickViewCardProps) {
  return (
    <Card className="border-border/70 bg-surface/95">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{data.ticker} Quick View</CardTitle>
            <CardDescription>
              {data.name} - {data.exchange}
            </CardDescription>
          </div>
          <Badge variant="positive">Bullish</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-col gap-1">
          <p className="text-3xl font-semibold text-text">{data.price}</p>
          <p className="text-sm text-positive">{data.weeklyChange}</p>
        </div>

        <div className="relative h-40 overflow-hidden rounded-xl border border-border/70 bg-panel/60">
          <div className="absolute inset-0 bg-gradient-to-br from-accent/15 via-transparent to-transparent" />
          <div className="absolute left-4 top-4 flex items-center gap-2 text-xs text-muted">
            <BarChart3 className="h-4 w-4" />
            Weekly price action
          </div>
          <svg
            className="absolute bottom-0 left-0 h-full w-full"
            viewBox="0 0 300 120"
            preserveAspectRatio="none"
          >
            <path
              d="M0,90 C40,70 60,85 90,60 C120,35 160,50 190,40 C220,30 250,35 300,15"
              fill="none"
              stroke="rgba(56, 241, 191, 0.85)"
              strokeWidth="2"
            />
            <path
              d="M0,90 C40,70 60,85 90,60 C120,35 160,50 190,40 C220,30 250,35 300,15 L300,120 L0,120 Z"
              fill="rgba(56, 241, 191, 0.08)"
            />
          </svg>
        </div>

        <div className="flex flex-wrap gap-2">
          <Button className="gap-2" size="md">
            Full Analysis
            <ArrowUpRight className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="md">
            Compare
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
