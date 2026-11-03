---
updated: 2026-09-30
---
# Project Structure

- `src/loans`: lending logic and the loan state machine.
- `src/borrowers`: borrower records.
- `src/theme`: design tokens.
- `migrations`: database migrations.

## Rule

Business logic stays in services. Routes only validate input and call a service.
