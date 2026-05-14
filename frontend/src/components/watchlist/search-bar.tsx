import { Filter, Plus, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type SearchBarProps = {
  value: string;
  onChange: (value: string) => void;
  onFilterClick: () => void;
  onAddClick: () => void;
};

export function SearchBar({ value, onChange, onFilterClick, onAddClick }: SearchBarProps) {
  return (
    <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
      <div className="relative w-full lg:max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <Input
          placeholder="Search watchlist..."
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="pl-9"
        />
      </div>
      <div className="flex w-full gap-2 sm:w-auto">
        <Button variant="outline" size="sm" className="gap-2" onClick={onFilterClick}>
          <Filter className="h-4 w-4" />
          Filter
        </Button>
        <Button size="sm" className="gap-2" onClick={onAddClick}>
          <Plus className="h-4 w-4" />
          Add Ticker
        </Button>
      </div>
    </div>
  );
}
