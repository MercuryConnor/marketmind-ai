import { ArrowDownRight, ArrowUpRight, MoreVertical } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { WatchlistRow } from "@/data/watchlist";

type WatchlistTableProps = {
  rows: WatchlistRow[];
};

export function WatchlistTable({ rows }: WatchlistTableProps) {
  return (
    <div className="rounded-2xl border border-border/70 bg-surface/90">
      <div className="border-b border-border/70 px-5 py-4">
        <h2 className="text-sm font-semibold text-text">Core Watchlist</h2>
        <p className="text-xs text-muted">Institutional names with live AI signals.</p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-[940px] w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-[0.2em] text-muted">
            <tr>
              {[
                "Ticker",
                "Company Name",
                "Price",
                "Change",
                "Market Cap",
                "P/E Ratio",
                "AI Sentiment",
                "Action",
              ].map((label) => (
                <th key={label} className="px-5 py-3 font-medium">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            {rows.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-5 py-6 text-sm text-muted">
                  No matches found. Try a different ticker or company name.
                </td>
              </tr>
            ) : (
              rows.map((row) => {
                const isPositive = row.changePct >= 0;
                return (
                  <tr
                    key={row.ticker}
                    className="group/row transition-colors hover:bg-panel/60"
                  >
                    <td className="border-l-2 border-transparent px-5 py-3 group-hover/row:border-accent">
                      <div className="flex items-center gap-3">
                        <span className="rounded-lg bg-panel px-2 py-1 text-xs font-semibold text-text">
                          {row.ticker}
                        </span>
                      </div>
                    </td>
                    <td className="px-5 py-3 text-text">{row.name}</td>
                    <td className="px-5 py-3 text-text">{row.price}</td>
                    <td className={cn("px-5 py-3", isPositive ? "text-positive" : "text-negative")}
                    >
                      <div className="flex items-center gap-1">
                        {isPositive ? (
                          <ArrowUpRight className="h-4 w-4" />
                        ) : (
                          <ArrowDownRight className="h-4 w-4" />
                        )}
                        {isPositive ? "+" : ""}
                        {Math.abs(row.changePct).toFixed(2)}%
                      </div>
                    </td>
                    <td className="px-5 py-3 text-text">{row.marketCap}</td>
                    <td className="px-5 py-3 text-text">{row.peRatio}</td>
                    <td className="px-5 py-3">
                      <Badge
                        variant={
                          row.sentiment === "Bullish"
                            ? "positive"
                            : row.sentiment === "Bearish"
                              ? "negative"
                              : "neutral"
                        }
                      >
                        {row.sentiment}
                      </Badge>
                    </td>
                    <td className="px-5 py-3">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="opacity-0 transition-opacity group-hover/row:opacity-100"
                            aria-label="Row actions"
                          >
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>View details</DropdownMenuItem>
                          <DropdownMenuItem>Compare</DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem className="text-negative">Remove</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
