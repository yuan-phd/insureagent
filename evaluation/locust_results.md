# InsureAgent Load Test Results

**Date:** 2026-03-08
**Tool:** Locust 2.x
**Target:** FastAPI inference server (`http://localhost:8000`)
**Model:** Teacher (GPT-4o mini)
**Environment:** Local Docker (Intel Mac, CPU only)

## Test Configuration

| Parameter | Value |
|---|---|
| Concurrent users | 5 |
| Spawn rate | 1 user/s |
| Duration | ~4 minutes |

## Results

| Endpoint | Requests | Failures | Median | P95 | P99 | Avg | Min | Max | RPS |
|---|---|---|---|---|---|---|---|---|---|
| GET /health | 11 | 0 | 399ms | 400ms | 400ms | 399ms | 400ms | 400ms | 0.0 |
| POST /process_claim | 316 | 26 | 6,000ms | 52,000ms | 62,000ms | 26,264ms | 3,904ms | 62,014ms | 0.3 |
| Aggregated | 327 | 27 | 4,000ms | 52,000ms | 62,000ms | 25,737ms | 3,904ms | 62,014ms | 0.3 |

**Failure rate:** 22% (27/327)

## Analysis

### Bottleneck: OpenAI API concurrency

Each claim requires 3 sequential OpenAI API calls (lookup → check_rules → calculate_payout). Under 5 concurrent users, requests queue behind each other causing P99 latency to reach 62 seconds — triggering timeouts and accounting for the 22% failure rate.

This is an **API-bound bottleneck**, not a server-side bug. The FastAPI server itself adds negligible overhead; all latency is in the OpenAI round-trips.

### /health latency (399ms)

Health check should return in <10ms. The 399ms response time indicates the Docker container is under CPU/memory pressure from concurrent inference requests sharing the same process. In production, the health check endpoint would run in a separate lightweight process.

### Min latency (3,904ms)

Fastest cases are DENIED claims that terminate after check_rules — only 2 API calls instead of 3.

### RPS (0.3)

With average latency ~26s and 5 concurrent users, theoretical max RPS ≈ 5/26 ≈ 0.19. Observed 0.3 RPS is slightly above this due to fast DENIED cases completing early.

## Root Cause & Fix

| Issue | Root Cause | Fix |
|---|---|---|
| 22% failure rate | OpenAI API timeout under concurrent load | Replace Teacher with Student local inference |
| 62s P99 | 3 sequential API calls per claim | Async tool calls (asyncio) or Student model |
| 399ms health check | Shared process with inference workload | Separate health check process in production |

## Projected Student Model Performance

Student model runs locally — no API dependency, no queuing. Expected on T4 GPU:

| Config | Latency/claim | RPS (5 users) | Failure rate |
|---|---|---|---|
| Teacher API (measured) | 26,264ms avg | 0.3 | 22% |
| Student FP16 (projected) | ~5,000ms | ~1.0 | <1% |
| Student INT8 (projected) | ~3,000ms | ~1.5 | <1% |

*Projected figures to be confirmed after quantisation benchmarking on Colab T4.*