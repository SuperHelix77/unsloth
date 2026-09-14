# Unsloth Helix: Qwen3.8-27B Mac profile

> **Beta / experimental**
>
> This document describes the current Unsloth Helix development profile. It is not a universal performance guarantee. Results depend on hardware, quantization, context length, backend, and speculative-draft compatibility.

Unsloth Helix turns Studio into a local, Codex-shaped harness for long-context inference, persistent agent work, governed learning, and reversible model adaptation. Existing Unsloth training and inference paths remain available; Helix features are additive and are expected to fall back cleanly when a model or backend cannot support them.

## Runtime profile

The primary optimized target is **Qwen3.8-27B on Apple Silicon**.

When relevant settings remain automatic, the current Helix Mac profile targets:

- 64K context when the model and available unified memory permit it;
- unified-memory-aware KV handling;
- KV `q4_0` defaults for the target profile;
- single-slot interactive decode;
- larger prefill batches where supported;
- a target-matched DFlash2 drafter when the selected backend can load and verify it.

Explicit user settings win. Helix policy must not silently replace an explicit context length, cache type, slot count, placement decision, or speculative-decoding mode.

If a speculative sidecar cannot load, the model should remain usable through ordinary decoding and the fallback should be visible. The application must not report an acceleration mode as active merely because it was requested.

DFlash-family and MTP paths are speculative-decoding accelerators, not alternate sources of truth: accepted draft tokens are verified by the target model before commitment. Observed speedup varies with the model pair, quantization, context, acceptance rate, backend, and hardware.

The 64K policy is applied at load resolution and in the relevant speculative-cache path so enabling a compatible speculative mode does not silently restore a smaller legacy cache window on a later turn.

## Learning ladder

Helix separates different kinds of learning instead of sending every failure directly into fine-tuning:

`experience → memory → repeated pattern → skill → proven model deficiency → QLoRA`

A more complete routing policy is:

- **memory** for durable facts, project state, and episodic context;
- **skill** for repeatable procedures that can be written explicitly;
- **ephemeral skill** for a procedure needed immediately but not yet proven durable;
- **runtime repair** when infrastructure caused the failure;
- **QLoRA challenger** for a recurring model-level behavior gap that survives good context and skills;
- **nothing** for noise, sampling variance, or one-off cases.

Completed local text turns are recorded in a bounded task ledger. Mem0 retrieval is supplementary and can be used to identify recurrence. A procedural repetition should normally become a plain-text skill before an expensive model-weight update is considered.

## Self-QLoRA acceptance gate

The promotion decision is external to the candidate model.

Current policy is designed around the following invariants:

- score the current baseline/champion and candidate on a fixed held-out benchmark;
- keep promotion data separate from QLoRA training data;
- require a measurable quality improvement under the configured benchmark policy;
- measure throughput and reject material performance regressions under that policy;
- keep the base/current champion available for comparison and rollback;
- hot-swap only a candidate that passed the gate;
- keep failed candidates inactive unless deliberately retrained or re-evaluated.

The model's own critique, confidence, or self-score is never sufficient promotion evidence.

Because adapters remain separate from the base checkpoint, promotion can be reversible rather than a destructive rewrite of the model.

## On-the-fly skill creation

Helix can draft a procedure while solving a task and later decide whether that procedure deserves durable promotion.

The intended lifecycle is:

`novel workflow → ephemeral skill draft → use/test → recurrence/evidence → durable skill or discard`

Generated public skills are plain-text `SKILL.md` artifacts. The skill path should not silently install arbitrary generated executable code.

Durable skills should retain enough provenance to answer why they were created and what evidence supported their promotion.

## Local Mem0

Mem0 is optional at import time and configured for account-scoped local storage:

- Qdrant files and history live under the account's `learning/mem0` directory;
- the vector-store user id is a one-way account hash rather than an email/account subject;
- embeddings use a local Hugging Face model;
- the Mem0 LLM points to Studio's local OpenAI-compatible endpoint by default;
- Mem0 and Hugging Face telemetry are disabled by the adapter where supported;
- if Mem0 is unavailable, the bounded JSON/task ledger continues to work.

Install the optional dependency in a compatible environment with:

```bash
pip install 'mem0ai>=0.1.118,<1.0'
```

The shipped runtime may already contain the package. Helix does not intentionally downgrade the existing environment merely to satisfy an optional memory package. The `/api/memory` status endpoint reports availability and the local ledger remains the fallback.

## Goal, Plan, and workspace surfaces

Helix adds persistent execution structure around local chat:

- **Goal** tracks the concrete outcome;
- **Plan** tracks durable execution steps/progress;
- **Outputs** exposes produced artifacts/results;
- **Background** exposes background activity;
- **Sources** exposes relevant source/context surfaces;
- **Subagents** exposes delegated work where available.

The intended loop is:

`goal → inspect → plan → act → observe → verify → update → continue/stop`

The runtime should stop or ask when it reaches a permission boundary, unresolved ambiguity, or an action outside its configured authority.

## Skills and commands

The local skill layer can discover installed Codex/Claude-style `SKILL.md` files and dynamic commands.

Current command surfaces include:

- `/github [query]`
- `/plan [objective]`
- `/goal [objective]`
- `/speed`
- `/learn`
- `/skills`
- `/skill-name request`

`/github` uses configured/authenticated GitHub context when available instead of requiring repository URLs to be copied into every prompt.

Skill installation is data-oriented: skill files are treated as instructions. Repository scripts are not supposed to execute merely because a skill repository was installed.

## GitHub context

The bundled `unsloth-local-codex` plugin includes a read-only GitHub MCP connection for Codex-style repository inspection.

OAuth/credentials belong to the MCP host or configured connector. Do not copy tokens into the plugin, repository, prompts, skills, or logs.

For implementation and performance work, prefer primary repository evidence and retain commit/file provenance in notes when it materially supports a claim.

## Computer use

Computer use is currently **macOS-only and experimental**.

Available primitives include:

- screenshot;
- click;
- type;
- key press;
- scroll;
- open application.

Screenshot is read-only. Mutating actions require approval under the local permission policy and use bounded native macOS automation surfaces.

Enable Accessibility for Unsloth Studio in:

`System Settings → Privacy & Security → Accessibility`

A permission failure should be reported to the model/user rather than silently bypassed.

## Performance evidence

Helix separates ordinary target decode from speculative decode.

A speculative speed claim is evidence-backed only when the runtime can show that:

- a speculative mode was actually engaged;
- the target/drafter pair loaded successfully;
- draft tokens were proposed;
- non-zero accepted draft tokens were verified and committed;
- measured wall time/throughput improved for the tested workload.

Useful metrics include:

- TTFT / time to first useful action;
- warm-turn latency;
- prefill/context-processing time;
- target decode tok/s;
- committed speculative tok/s;
- accepted draft tokens / acceptance length;
- tool-result-to-next-action latency;
- peak unified memory;
- completed-task wall time;
- task success and regression rate.

Do not generalize a result from one model, quant, context length, or Mac configuration to every installation.

## Beta boundaries

The current public Helix profile does **not** claim:

- universal speedup from DFlash/MTP;
- identical behavior across every Unsloth-supported model/backend;
- production-ready autonomous computer use;
- that a benchmark-passing QLoRA is globally more intelligent;
- that generated skills are automatically trustworthy;
- that optional memory is required for ordinary inference.

The primary rule is simple: preserve the working baseline, measure replacements against it, and keep rollback available.
