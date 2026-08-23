# HTTP API

## Authentication

Every request carries a workspace token in the Authorization header. Tokens are scoped
to a single workspace and never expire, but they can be revoked from the dashboard.

## Rate limits

The API allows 600 requests per minute per token. Exceeding the limit returns status
429 together with a Retry-After header giving the number of seconds to wait. Rate limit
counters reset on a sliding window, not on the minute boundary.

## Pagination

List endpoints return at most 100 items per page. Continue by passing the cursor from
the next_cursor field; offset based paging is not supported.

## Webhooks

A delivery that does not return 2xx is retried with exponential backoff: after 1
minute, then 5 minutes, then 30 minutes, then 2 hours, and finally after 6 hours. After
the fifth failed attempt the endpoint is disabled and the workspace owner is notified.
Payloads are signed with HMAC SHA-256 and the signature travels in the X-Northwind
-Signature header.
