# Early stopping strategy for LLM training

## Metadata

- **Conversation ID:** 95a2724d-cc61-4fb0-9d0a-7690e6fe2132
- **Mode:** agent
- **Created:** 2026-03-17 08:20:27
- **Last Updated:** 2026-03-17 08:52:27
- **Branch:** main
- **Files Changed:** 1
- **Lines Added:** 714
- **Lines Removed:** 0
- **Summary:** Edited 2026-03-17-early-stopping-for-continued-training.md

---

## Conversation

### 🧑 User

<manually_attached_skills>
The user has manually attached the following skills to their message.
These skills contain specific instructions or workflows that the user wants you to follow for this request.
Only read the files if needed, the full skill content is inlined here.

Skill Name: writing-plans
Path: /Users/a964286/.cursor/skills/writing-plans/SKILL.md
SKILL.md content:
# Writing Plans

## Overview

Write comprehensive implementation plans assuming the engineer has zero context for our codebase and questionable taste. Document everything they need to know: which files to touch for each task, code, testing, docs they might need to check, how to test it. Give them the whole plan as bite-sized tasks. DRY. YAGNI. TDD. Frequent commits.

Assume they are a skilled developer, but know almost nothing about our toolset or problem domain. Assume they don't know good test design very well.

**Announce at start:** "I'm using the writing-plans skill to create the implementation plan."

**Context:** This should be run in a dedicated worktree (created by brainstorming skill).

**Save plans to:** `docs/plans/YYYY-MM-DD-<feature-name>.md`

## Bite-Sized Task Granularity

**Each step is one action (2-5 minutes):**
- "Write the failing test" - step
- "Run it to make sure it fails" - step
- "Implement the minimal code to make the test pass" - step
- "Run the tests and make sure they pass" - step
- "Commit" - step

## Plan Document Header

**Every plan MUST start with this header:**

```markdown
# [Feature Name] Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** [One sentence describing what this builds]

**Architecture:** [2-3 sentences about approach]

**Tech Stack:** [Key technologies/libraries]

---
```

## Task Structure

````markdown
### Task N: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Modify: `exact/path/to/existing.py:123-145`
- Test: `tests/exact/path/to/test.py`

**Step 1: Write the failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/path/test.py::test_name -v`
Expected: FAIL with "function not defined"

**Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/path/test.py::test_name -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```
````

## Remember
- Exact file paths always
- Complete code in plan (not "add validation")
- Exact commands with expected output
- Reference relevant skills with @ syntax
- DRY, YAGNI, TDD, frequent commits

## Execution Handoff

After saving the plan, offer execution choice:

**"Plan complete and saved to `docs/plans/<filename>.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?"**

**If Subagent-Driven chosen:**
- **REQUIRED SUB-SKILL:** Use superpowers:subagent-driven-development
- Stay in this session
- Fresh subagent per task + code review

**If Parallel Session chosen:**
- Guide them to open new session in worktree
- **REQUIRED SUB-SKILL:** New session uses superpowers:executing-plans
</manually_attached_skills>
You are an expert in LLM pretraining and continuous training; in the section continue to learn @dev/legacy/legacy_full_training.ipynb, I would like to implement a early stop strategy; here is the pre-research before planning and here is the training results of the first epoch training @expe_logs/exp_round5/exp1_dbcheck/; /writing-plans

---

### 🤖 Assistant

I'm using the writing-plans skill to create the implementation plan.

Let me start by gathering all the context I need — the pre-research document, the training notebook, and the experiment results.

---

### 🤖 Assistant

Now let me read the relevant sections of the notebook — specifically the "Continue to learn" section and the existing training infrastructure.

---

### 🤖 Assistant

Now let me also read the `train_epoch` function and the rest of the continue-training function to see the full picture.

---

### 🤖 Assistant

Now let me check the `save_checkpoint_local` function signature and the `evaluate` function's `max_batches` parameter, since partial validation is a key part of the plan.

---

### 🤖 Assistant

Now I have a thorough understanding of the full system. Let me check one more thing — the `evaluate` function already supports `max_batches` for partial validation, which is crucial for the sub-epoch validation strategy.

Now let me write the comprehensive implementation plan.

---

### 🤖 Assistant

Plan complete and saved to `docs/plans/2026-03-17-early-stopping-for-continued-training.md`.

Here's a summary of what the plan covers across **7 tasks**:

### Architecture

A **composite sub-epoch early stopping strategy** with three integrated components:

1. **`EarlyStoppingConfig`** — a `@dataclass` holding all configuration (metric, patience, warmup, check interval, val fraction, etc.)
2. **`EarlyStoppingMonitor`** — the stateful tracker that records validation history, detects improvement/plateau, and decides when to stop
3. **`on_optimizer_step` callback** — a new optional parameter on `train_epoch()` that fires after every optimizer step, enabling mid-epoch validation without rewriting the training loop

### Key Design Decisions (from your pre-research)

- **Primary metric: NDCG@20** — not val_loss, because BCE across 6,297 targets is dominated by negatives; NDCG directly measures clinical code ranking quality
- **Sub-epoch validation every 500 optimizer steps** (~1.2 hours) — gives 6 checkpoints per epoch instead of 1
- **Partial validation (20%)** — ~12 min per check instead of ~1 hour, using the existing `evaluate(max_batches=...)` parameter
- **Warmup of 1000 steps** — protects the CosineAnnealingLR exploration phase
- **Patience of 5 checks** — ~2,500 steps of stagnation before stopping, balancing noise tolerance with responsiveness
- **Best-model restoration** — automatically loads the best checkpoint when early stopping triggers
- **Full backward compatibility** — the original training loop (Section 6) is unaffected; the callback defaults to `None`

### Execution Options

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open a new session with the executing-plans skill, batch execution with checkpoints

Which approach would you prefer?

