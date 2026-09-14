---
name: unsloth-local-codex
description: Use Unsloth Helix as a local Codex-like coding, planning, and inference workspace with durable Goal/Plan mode, GitHub context, long-context Mac safeguards, skills, memory, and evidence-based performance changes.
---

# Unsloth Helix Local Codex

Use this skill when working on Unsloth Helix or when the user wants a local Codex-style workflow backed by Unsloth Studio.

> Helix is a beta/research surface. Keep changes measurable, reversible, and explicit about fallbacks.

## Operating rules

- Keep **Goal** mode for the concrete outcome and **Plan** mode for the durable execution plan. Preserve both when reopening or switching chat threads.
- Inspect the relevant repository evidence before applying an optimization. Treat benchmark claims as scoped evidence with model, hardware, context, quantization, and runtime conditions; do not promote a best-case result to a universal promise.
- Keep explicit user settings authoritative. Automatic policies may choose safe defaults, but must never silently overwrite an explicit context length, cache type, slot count, speculative mode, or placement choice.
- For the primary Mac Qwen3.8-27B profile, Helix targets 64K context when automatic settings and available unified memory permit it. A fit-reduced window is a runtime outcome, not user intent, and must not become a sticky request on the next reload.
- MTP and DFlash-family modes are experimental unless the target/drafter pair and installed backend capability agree. If a drafter fails, preserve ordinary target decoding and report the fallback clearly.
- Do not label an acceleration mode active merely because it was requested. Performance reports require runtime evidence such as an active speculative lane and accepted/committed draft tokens.
- Prefer **memory** for facts, **skills** for repeatable procedures, **runtime repair** for infrastructure defects, and **QLoRA** only for recurring model-level behavior gaps that survive good context and skills.
- Treat learned QLoRA adapters as challengers. Promotion requires the external benchmark gate; the model's own confidence or critique is not sufficient evidence.
- Keep generated skills inspectable and reversible. Skill installation must not execute arbitrary repository scripts merely because a skill was discovered.

## GitHub context

In the desktop app, the visible Skills menu and these local commands are available where the corresponding feature is enabled:

- `/github [query]` resolves authenticated GitHub/local repository context without requiring the user to paste a URL for every task.
- `/plan [objective]` selects durable Plan mode.
- `/goal [objective]` selects durable Goal mode.
- `/speed` reports measured runtime information such as tok/s, context length, and whether speculative decoding is actually engaged.
- `/learn` opens/invokes the learning workflow.
- `/skills` shows the command/skill menu.
- `/skill-name request` injects the selected bounded `SKILL.md` instructions into the local request.

The Skills manager can create a portable `SKILL.md` or install a read-only skill repository into supported local skill locations. Skill files are instructions/data for the local runtime; do not execute repository scripts during installation.

Use the bundled read-only GitHub MCP connection for repository, issue, pull-request, and source inspection when running in Codex. OAuth is handled by the MCP host; do not copy tokens into the plugin, repository, prompts, skills, or logs.

Prefer primary GitHub source and preserve commit/file provenance in implementation notes when it materially supports a decision.

## Learning policy

When an episode suggests the system should learn something, classify it before changing durable state:

1. **Nothing** — one-off noise, sampling variance, or insufficient evidence.
2. **Memory** — durable fact, project state, prior result, or episodic context.
3. **Ephemeral skill** — a novel procedure useful now but not yet proven recurrent.
4. **Durable skill** — a repeatable, explicit procedure with evidence of reuse.
5. **Runtime repair** — infrastructure/tool/backend behavior caused the failure.
6. **QLoRA challenger** — a recurring model-level behavior gap not cleanly solved by context or skills.

A QLoRA candidate must remain inactive until it passes the configured held-out comparison against the current baseline/champion. Preserve rollback.

## Computer use

Computer use is currently experimental and macOS-focused.

- Screenshot is read-only.
- Click, type, key, scroll, and app-opening actions are permission-gated.
- Respect macOS Accessibility boundaries; report permission failures rather than bypassing them.
- Do not use GUI automation when a safer, deterministic local tool/API is already available for the same operation.

## Delivery checklist

1. Identify the concrete goal, current plan, relevant repository state, and permission boundary.
2. Establish the current baseline before replacing or optimizing a module.
3. Add the smallest policy/implementation seam that can be independently tested and disabled.
4. Verify backend/frontend tests relevant to the change plus formatting/type/build checks as appropriate.
5. Measure the replacement against the baseline on the same workload.
6. Report fallbacks and unsupported paths explicitly; do not count requested-but-inactive acceleration as success.
7. Keep learned skills/adapters reversible and preserve the prior champion/baseline.
8. Build the target app before claiming the installed application changed.
9. Report physical unified-memory limits separately from software policy limits.
