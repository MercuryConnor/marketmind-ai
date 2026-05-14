import {
  Bookmark,
  History,
  Info,
  LayoutGrid,
  LineChart,
  MessageSquareText,
  Settings,
  Star,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";

const navItems = [
  { label: "Dashboard", icon: LayoutGrid },
  { label: "Ask MarketMind", icon: MessageSquareText },
  { label: "Market Overview", icon: LineChart },
  { label: "Watchlist", icon: Star, active: true },
  { label: "Saved Insights", icon: Bookmark },
  { label: "History", icon: History },
  { label: "Settings", icon: Settings },
];

type SidebarProps = {
  isMobileOpen: boolean;
  onClose: () => void;
};

function SidebarContent({ onClose }: { onClose?: () => void }) {
  return (
    <div className="flex h-full flex-col gap-6 px-4 py-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/15 text-sm font-semibold text-accent">
            MM
          </div>
          <div>
            <p className="text-sm font-semibold text-text">MarketMind</p>
            <p className="text-xs text-muted">Financial AI Assistant</p>
          </div>
        </div>
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-border bg-panel/70 p-1.5 text-muted hover:text-text"
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      <nav className="flex-1 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.label}
              type="button"
              aria-current={item.active ? "page" : undefined}
              className={cn(
                "flex w-full items-center gap-3 rounded-xl border border-transparent px-3 py-2 text-sm text-muted transition-colors",
                "hover:bg-panel/60 hover:text-text",
                item.active &&
                  "border-border bg-panel/80 text-accent border-l-2 border-l-accent"
              )}
            >
              <Icon className="h-4 w-4" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="space-y-4">
        <div className="rounded-xl border border-border bg-panel/60 p-3">
          <p className="text-xs uppercase tracking-[0.2em] text-muted">Market Status</p>
          <div className="mt-2 flex items-center gap-2 text-sm text-text">
            <span className="h-2 w-2 rounded-full bg-accent" />
            All Systems Operational
          </div>
        </div>
        <button
          type="button"
          className="flex items-center gap-2 text-xs text-muted transition-colors hover:text-text"
        >
          <Info className="h-3.5 w-3.5" />
          About MarketMind
        </button>
      </div>
    </div>
  );
}

export function Sidebar({ isMobileOpen, onClose }: SidebarProps) {
  return (
    <>
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[260px] border-r border-border/70 bg-surface/95 lg:flex">
        <SidebarContent />
      </aside>

      <div className={cn("lg:hidden", isMobileOpen ? "block" : "hidden")}>
        <div className="fixed inset-0 z-40 bg-black/60" onClick={onClose} />
        <aside className="fixed inset-y-0 left-0 z-50 w-[260px] border-r border-border/70 bg-surface/95">
          <SidebarContent onClose={onClose} />
        </aside>
      </div>
    </>
  );
}
