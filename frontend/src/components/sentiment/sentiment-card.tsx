import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

type SentimentCardProps = {
  data: {
    insight: string;
    support: string;
    resistance: string;
    outlook: string;
  };
};

export function SentimentCard({ data }: SentimentCardProps) {
  return (
    <Card className="border-border/70 bg-surface/95 border-l-4 border-l-accent/70">
      <CardHeader>
        <CardTitle>AI Sentiment Insight</CardTitle>
        <CardDescription>NVDA momentum outlook</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 text-sm text-text">
        <p className="text-muted">{data.insight}</p>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-[0.2em] text-muted">Support</span>
            <span className="font-semibold text-text">{data.support}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-[0.2em] text-muted">Resistance</span>
            <span className="font-semibold text-text">{data.resistance}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-[0.2em] text-muted">Outlook</span>
            <Badge variant="positive">{data.outlook}</Badge>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
