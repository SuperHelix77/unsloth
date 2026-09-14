// SPDX-License-Identifier: AGPL-3.0-only

import assert from "node:assert/strict";
import test from "node:test";

import {
  hermesLearningReviewPrompt,
  isHermesLearningReviewRequest,
  parseOnTheFlySkillDraft,
  parseHermesLearningProposal,
  stripOnTheFlySkillDraft,
  stripHermesLearningProposal,
} from "../src/features/chat/lib/hermes-learning.ts";

test("Hermes review prompt is detectable and uses the staged proposal protocol", () => {
  const prompt = hermesLearningReviewPrompt("the loader fix");
  assert.equal(isHermesLearningReviewRequest(prompt), true);
  assert.match(prompt, /unsloth-learning/);
  assert.match(prompt, /Do not claim that anything was saved/);
});

test("learning marker is parsed and removed from the visible answer", () => {
  const answer = `Verified result.\n<unsloth-learning>{"kind":"memory","title":"Keep checks","content":"Run the focused test after changes.","reason":"Reusable verification habit."}</unsloth-learning>`;
  assert.deepEqual(parseHermesLearningProposal(answer), {
    kind: "memory",
    title: "Keep checks",
    content: "Run the focused test after changes.",
    reason: "Reusable verification habit.",
  });
  assert.equal(stripHermesLearningProposal(answer), "Verified result.");
});

test("on-the-fly skill drafts are parsed, bounded, and hidden from the answer", () => {
  const answer = `Completed the task.\n<unsloth-skill-draft>{"name":"json-checks","title":"Verify JSON contracts","content":"Compare parsed values against the contract.","reason":"The same procedure recurred."}</unsloth-skill-draft>`;
  assert.deepEqual(parseOnTheFlySkillDraft(answer), {
    kind: "skill",
    name: "json-checks",
    title: "Verify JSON contracts",
    content: "Compare parsed values against the contract.",
    target: "codex",
    recommendationAction: "skill",
    reason: "The same procedure recurred.",
    recommendationReason:
      "A repeatable procedure was identified; stage it as a skill before considering QLoRA.",
  });
  assert.equal(stripOnTheFlySkillDraft(answer), "Completed the task.");
});

test("learning recommendations accept runtime repair without turning it into a score", () => {
  const proposal = parseHermesLearningProposal(
    '<unsloth-learning>{"kind":"memory","title":"Runtime repair","content":"Revert a failing adapter before retraining.","recommendationAction":"runtime-fix","recommendationReason":"The active candidate failed."}</unsloth-learning>',
  );
  assert.equal(proposal?.recommendationAction, "runtime-fix");
  assert.equal(proposal?.recommendationReason, "The active candidate failed.");
});
