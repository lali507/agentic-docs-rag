---
updated: 2026-09-18
---
# Design Decisions

## Primary color

Date: 2026-07-22

We started with blue `#1F6FEB` as the primary color.

## Primary color update

Date: 2026-09-18

The primary color was changed to teal `#0F766E` because the blue did not meet contrast requirements on the printed loan slips. This is the current primary color.

## Interface languages

Date: 2026-09-02

The interface text will be translated into Hebrew, English and Yiddish. Hebrew is the default language.

## RTL scope

Date: 2026-08-05

RTL is applied only on the main screens for now. Admin screens stay LTR until the first release.

## SMS reminders

The SMS provider plan is capped at 100 messages per day, so reminders must be batched.
