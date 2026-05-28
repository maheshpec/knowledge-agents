# Orion planner

Orion is the planning service. It takes a goal and a context bundle and emits
a step-by-step plan that the orchestrator executes one step at a time. Each
step names the tool to invoke and the inputs to feed it.

Orion submits its model calls through the Falcon engine like every other
caller; this is where the harness keeps a single chokepoint for inference
budget and observability. Orion does not maintain its own LLM connection pool.

Plans are versioned: when adaptation rewrites a step, the previous version is
kept so reviewers can audit the trajectory.

The planner caps its own work via `ORION_MAX_PLAN_STEPS = 32`, refusing to
extend a plan beyond that ceiling so a runaway adaptation can't burn unlimited
budget.
