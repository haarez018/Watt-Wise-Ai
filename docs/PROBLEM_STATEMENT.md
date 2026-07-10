# Problem statement

## The problem

An Indian household gets an electricity bill once a month. By the time it arrives, the
consumption it charges for is already locked in — there's no way to have acted on it
sooner. The bill also gives almost no insight into *why* it's the number it is: which
appliance drove the increase, whether this month is actually unusual for the season, or
what a specific, quantified action would save.

Meanwhile India's grid is roughly 75% coal-dependent (per the Central Electricity
Authority's baseline data). Every kWh a household doesn't consume is, overwhelmingly,
coal that wasn't burned. Household electricity efficiency is one of the few sustainability
levers an individual can pull that has an immediate, measurable, personally-relevant
payoff — lower bill *and* lower emissions, not a tradeoff between them.

## Why existing options fall short

- **Smart plugs / hardware NILM:** accurate, but requires buying and installing
  hardware per appliance — a real barrier for most households, and not something a
  renter or a family with an old wiring setup will do.
- **Utility apps:** show the same bill data back, rarely a forecast, essentially never
  an appliance breakdown, and never CO₂.
- **Generic budgeting apps:** treat electricity like any other expense line item, with
  no domain model of tariffs, appliances, or seasonality.

## What WattWise AI does differently

1. **Forecasts next month's bill** before it arrives, with a confidence interval, from
   the household's own bill history — so a spike is visible before the money is spent,
   not after.
2. **Detects anomalies** in plain language ("this month is 40% above typical for this
   season") instead of a bare number.
3. **Disaggregates the bill by appliance category without any hardware** — a
   software model estimates what the fridge, AC, geyser, lighting, fans, washing
   machine, and standby load are each contributing, from the household's appliance
   inventory and total consumption.
4. **Quantifies every recommendation in both ₹/month and kg CO₂/year**, with the
   calculation method inspectable, not a black box.
5. **Tracks realized savings over time**, so a recommendation isn't just suggested —
   its actual impact is proven back to the user.

## User friction budget

Two things, once: create an account, and enter 6–12 months of past bills plus a
one-time appliance inventory (~60 seconds of checkboxes with age and star rating).
After that, the only ongoing action is entering (or uploading) each new month's bill —
a single number. No photos, no hardware, no daily logging.

## What "done" looks like for this hackathon submission

A publicly deployed, multi-tenant web application — not a local demo — where a real
household could sign up today, enter their bill history, and get a forecast, an
appliance breakdown, and ranked recommendations, all computed by models we trained
ourselves, with zero external AI calls on the request path. Production security,
observability, and data-handling standards apply from the first commit, because the
intent is for this to survive contact with real users past the hackathon, not just the
demo.

## Current status

Phase 1 (production scaffolding: monorepo, auth, base schema, CI, observability,
this documentation) is complete. Phases 2–5 (ML training pipeline, backend business
endpoints, frontend dashboard UX, and launch readiness) are designed in
`ARCHITECTURE.md`, `API.md`, and `ML.md` and build on this foundation in order.
