import { Bell, Menu, UserCircle2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

type TopHeaderProps = {
  onOpenNav: () => void;
};

export function TopHeader({ onOpenNav }: TopHeaderProps) {
  return (
    <header className="sticky top-0 z-30 border-b border-border/70 bg-background/80 backdrop-blur">
      <div className="flex items-center justify-between px-4 py-4 lg:px-8">
        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden"
            onClick={onOpenNav}
            aria-label="Open navigation"
          >
            <Menu className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-lg font-semibold text-text">Your Watchlist</h1>
            <p className="text-xs text-muted">Live market intelligence</p>
          </div>
          <div className="ml-4 hidden items-center gap-2 rounded-full border border-border/70 bg-panel/60 px-3 py-1 text-xs text-muted md:flex">
            <span className="h-2 w-2 rounded-full bg-accent" />
            Live Market
            <span className="rounded-full bg-accent/20 px-2 py-0.5 text-[10px] font-semibold text-accent">
              ON
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Button variant="ghost" size="icon" aria-label="Notifications">
            <Bell className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" aria-label="Profile">
            <UserCircle2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </header>
  );
}
