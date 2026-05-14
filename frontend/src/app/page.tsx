"use client";

import * as React from "react";

import { QuickViewCard } from "@/components/dashboard/quick-view-card";
import { Sidebar } from "@/components/layout/sidebar";
import { TopHeader } from "@/components/layout/top-header";
import { SentimentCard } from "@/components/sentiment/sentiment-card";
import { SearchBar } from "@/components/watchlist/search-bar";
import { WatchlistTable } from "@/components/watchlist/watchlist-table";
import { quickView, watchlistRows } from "@/data/watchlist";

export default function Home() {
  const [query, setQuery] = React.useState("");
  const [isMobileNavOpen, setIsMobileNavOpen] = React.useState(false);

  const filteredRows = React.useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
      return watchlistRows;
    }
    return watchlistRows.filter((row) =>
      `${row.ticker} ${row.name}`.toLowerCase().includes(normalized)
    );
  }, [query]);

  const handleFilterClick = React.useCallback(() => null, []);
  const handleAddClick = React.useCallback(() => null, []);

  return (
    <div className="min-h-screen bg-background text-text">
      <Sidebar isMobileOpen={isMobileNavOpen} onClose={() => setIsMobileNavOpen(false)} />

      <div className="lg:pl-[260px]">
        <TopHeader onOpenNav={() => setIsMobileNavOpen(true)} />

        <main className="px-4 pb-12 pt-6 lg:px-8">
          <div className="mx-auto flex max-w-[1600px] flex-col gap-6">
            <SearchBar
              value={query}
              onChange={setQuery}
              onFilterClick={handleFilterClick}
              onAddClick={handleAddClick}
            />

            <WatchlistTable rows={filteredRows} />

            <section className="grid gap-6 xl:grid-cols-[2fr_1fr]">
              <QuickViewCard data={quickView} />
              <SentimentCard data={quickView} />
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
