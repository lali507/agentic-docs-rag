---
updated: 2026-10-12
---
# Tech Stack

## Decisions

- Decision (2026-06-28): the frontend uses React with TypeScript and Vite.
- Decision (2026-06-28): the backend uses Fastify.
- Decision (2026-07-10): the database is SQLite through Drizzle ORM.
- Decision (2026-10-12): Twilio replaces the previous SMS provider, keeping the same plan size.

## Constraints

The SMS plan allows at most 100 messages a day. SQLite allows a single writer at a time, so long imports must run outside active hours.

## Dependencies

Node.js 22 is required.
