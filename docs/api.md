# API Reference

## Base URL

- Local: http://127.0.0.1:8000

## Endpoints

### GET /health

Returns service status.

#### Response 200

```json
{
  "status": "running"
}
```

### POST /ask

Submits a financial query to the assistant.

#### Request Body

```json
{
  "query": "How did AAPL perform this week?"
}
```

#### Response 200

```json
{
  "analysis": "...",
  "data": {
    "query": "How did AAPL perform this week?",
    "market_data": {
      "symbol": "AAPL",
      "price": 190.2,
      "market_cap": 2900000000000.0,
      "volume": 54123000,
      "weekly_change": -1.35
    },
    "rag_context": {
      "query": "How did AAPL perform this week?",
      "snippets": [
        {
          "text": "...",
          "score": 0.0
        }
      ]
    },
    "errors": []
  },
  "insight": "..."
}
```

#### Error Responses

- 400 Bad Request
  - Empty query or invalid semantic input.
- 422 Unprocessable Entity
  - Invalid JSON schema.
- 500 Internal Server Error
  - Agent or tool execution failure.

#### Example cURL

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"What is P/E ratio?"}'
```

## Notes

- The response data object may include partial context when one tool fails.
- The agent falls back gracefully and still returns analysis/insight when possible.
