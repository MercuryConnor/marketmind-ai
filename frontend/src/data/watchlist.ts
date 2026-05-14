export type WatchlistSentiment = "Bullish" | "Neutral" | "Bearish";

export type WatchlistRow = {
  ticker: string;
  name: string;
  price: string;
  changePct: number;
  marketCap: string;
  peRatio: string;
  sentiment: WatchlistSentiment;
};

export const watchlistRows: WatchlistRow[] = [
  {
    ticker: "NVDA",
    name: "NVIDIA Corporation",
    price: "$924.79",
    changePct: 6.21,
    marketCap: "2.24T",
    peRatio: "66.21",
    sentiment: "Bullish",
  },
  {
    ticker: "AAPL",
    name: "Apple Inc.",
    price: "$189.98",
    changePct: 1.24,
    marketCap: "2.93T",
    peRatio: "29.45",
    sentiment: "Neutral",
  },
  {
    ticker: "TSLA",
    name: "Tesla, Inc.",
    price: "$173.44",
    changePct: -2.31,
    marketCap: "552.4B",
    peRatio: "42.10",
    sentiment: "Bearish",
  },
  {
    ticker: "MSFT",
    name: "Microsoft Corp.",
    price: "$430.12",
    changePct: 0.85,
    marketCap: "3.19T",
    peRatio: "37.82",
    sentiment: "Bullish",
  },
];

export const quickView = {
  ticker: "NVDA",
  name: "NVIDIA Corporation",
  exchange: "NASDAQ",
  price: "$924.79",
  weeklyChange: "+6.21% this week",
  support: "$880.00",
  resistance: "$950.00",
  outlook: "Strong Buy",
  insight:
    "Momentum remains constructive with buyers defending the $900 zone. Volume strength suggests the rally can extend if macro sentiment holds.",
};
