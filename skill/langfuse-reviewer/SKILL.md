---
name: langfuse-reviewer
description: >
  Review Langfuse traces, observations, scores and metrics to debug issues
  in yuno.md. Combines MCP tools (prompts, docs) with REST API calls (traces,
  observations, sessions, scores) for full observability access.
metadata:
  author: xuno-io
  version: "0.1.0"
  homepage: https://yuno.md
---

# Langfuse Reviewer

Review and debug yuno.md production traces directly from Claude Code.

## Available Tools

### MCP Tools (ready to use)

- **`langfuse` server** — Prompt management: `listPrompts`, `getPrompt`, `createTextPrompt`, `createChatPrompt`, `updatePromptLabels`
- **`langfuse-docs` server** — Documentation search: `searchLangfuseDocs`, `getLangfuseDocsPage`, `getLangfuseOverview`

### REST API (via curl)

The Langfuse REST API provides access to traces, observations, scores, sessions, and metrics. Authenticate with Basic Auth using credentials from `.env`.

## Authentication

All curl commands use Basic Auth. Read credentials from the project `.env` file:

```bash
source .env
LANGFUSE_AUTH="${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}"
LANGFUSE_API="${LANGFUSE_BASE_URL}/api/public"
```

## REST API Reference

### List Traces

```bash
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/traces?limit=10&orderBy=timestamp.desc" | jq .
```

Useful query parameters:
- `page` / `limit` — pagination (page starts at 1)
- `userId` — filter by user
- `name` — filter by trace name (e.g. `get_response`)
- `sessionId` — filter by session
- `fromTimestamp` / `toTimestamp` — ISO 8601 datetime range
- `orderBy` — format `field.asc|desc` (fields: `timestamp`, `name`, `userId`)
- `tags` — filter by tags (array)
- `fields` — comma-separated: `core`, `io`, `scores`, `observations`, `metrics`

### Get Single Trace

```bash
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/traces/{traceId}" | jq .
```

Returns full trace with input, output, metadata, observations, and scores.

### List Observations

```bash
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/observations?traceId={traceId}&limit=20" | jq .
```

Useful query parameters:
- `traceId` — get all observations for a specific trace
- `name` — filter by observation name
- `type` — filter by type (`SPAN`, `GENERATION`, `EVENT`)
- `level` — filter by level (`DEBUG`, `DEFAULT`, `WARNING`, `ERROR`)
- `fromStartTime` / `toStartTime` — ISO 8601 datetime range

### Get Single Observation

```bash
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/observations/{observationId}" | jq .
```

### List Scores

```bash
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/scores?limit=20" | jq .
```

Useful query parameters:
- `name` — filter by score name
- `userId` — filter by user
- `traceId` — filter by trace
- `source` — filter by source (`API`, `ANNOTATION`, `EVAL`)
- `dataType` — filter by type (`NUMERIC`, `CATEGORICAL`, `BOOLEAN`)

### List Sessions

```bash
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/sessions?limit=10" | jq .
```

### Metrics API

```bash
curl -s -u "$LANGFUSE_AUTH" -G \
  --data-urlencode 'query={
    "view": "observations",
    "metrics": [{"measure": "totalCost", "aggregation": "sum"}, {"measure": "count", "aggregation": "count"}],
    "dimensions": [{"field": "name"}],
    "filters": [],
    "fromTimestamp": "2025-01-01T00:00:00Z",
    "toTimestamp": "2026-12-31T00:00:00Z",
    "orderBy": [{"field": "totalCost_sum", "direction": "desc"}]
  }' \
  "${LANGFUSE_API}/v2/metrics" | jq .
```

## Debugging Workflows

### 1. Investigate a specific user issue

```bash
# Find recent traces for a user
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/traces?userId={telegramUserId}&limit=5&orderBy=timestamp.desc&fields=core,scores,metrics" | jq .

# Get the full trace with I/O
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/traces/{traceId}" | jq .

# Get observations (LLM calls, spans) within that trace
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/observations?traceId={traceId}" | jq .
```

### 2. Find errors

```bash
# Get observations with ERROR level
curl -s -u "$LANGFUSE_AUTH" \
  "${LANGFUSE_API}/observations?level=ERROR&limit=10" | jq '.data[] | {id, name, traceId, statusMessage, startTime}'
```

### 3. Cost analysis

```bash
# Total cost by model in the last 7 days
curl -s -u "$LANGFUSE_AUTH" -G \
  --data-urlencode 'query={
    "view": "observations",
    "metrics": [{"measure": "totalCost", "aggregation": "sum"}, {"measure": "count", "aggregation": "count"}],
    "dimensions": [{"field": "providedModelName"}],
    "filters": [],
    "fromTimestamp": "'"$(date -d '7 days ago' -Iseconds)"'",
    "toTimestamp": "'"$(date -Iseconds)"'",
    "orderBy": [{"field": "totalCost_sum", "direction": "desc"}]
  }' \
  "${LANGFUSE_API}/v2/metrics" | jq .
```

### 4. Latency analysis

```bash
# p95 latency by trace name
curl -s -u "$LANGFUSE_AUTH" -G \
  --data-urlencode 'query={
    "view": "observations",
    "metrics": [{"measure": "latency", "aggregation": "p95"}, {"measure": "latency", "aggregation": "avg"}],
    "dimensions": [{"field": "name"}],
    "filters": [],
    "fromTimestamp": "'"$(date -d '7 days ago' -Iseconds)"'",
    "toTimestamp": "'"$(date -Iseconds)"'",
    "orderBy": [{"field": "latency_p95", "direction": "desc"}]
  }' \
  "${LANGFUSE_API}/v2/metrics" | jq .
```

### 5. Review prompts

Use the MCP tools directly:

```
# List all prompts
mcp__langfuse__listPrompts(page=1, limit=50)

# Get production version of a prompt
mcp__langfuse__getPrompt(name="prompt-name")

# Search docs for a concept
mcp__langfuse-docs__searchLangfuseDocs(query="how to filter traces by metadata")
```

## Tips

- Always start by sourcing `.env` to get `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`
- Use `jq` to filter large responses: `| jq '.data[:3]'` for first 3 results
- Use `fields=core,metrics` on traces to skip heavy I/O data when you only need metadata
- The `filter` query parameter accepts JSON arrays and overrides individual query params when both are provided
- Trace IDs from Langfuse can be correlated with the `@observe()` decorated methods in `app/services/NeibotService/neibot_service.py`
- Sessions in Langfuse map to conversation threads in yuno.md
- When investigating a Linear issue, check if the issue mentions a user ID or trace ID to narrow the search
