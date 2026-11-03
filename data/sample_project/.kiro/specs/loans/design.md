---
updated: 2026-10-14
---
# Design: Loans

## Loan states

A loan moves through `active`, `overdue` and `returned`. All transitions go through `src/loans/stateMachine.ts`.

## Warning

Do not update `loans.status` directly from a route or a script. It bypasses the state machine and corrupts the inventory count.
