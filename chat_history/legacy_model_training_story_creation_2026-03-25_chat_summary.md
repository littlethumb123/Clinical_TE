# Legacy model training story creation

## Metadata

- **Conversation ID:** 51793bfe-b648-414a-b4c8-f2becd93b0e5
- **Mode:** agent
- **Created:** 2026-03-25 10:48:14
- **Last Updated:** 2026-03-25 10:49:08
- **Branch:** main
- **Files Changed:** 2
- **Lines Added:** 17
- **Lines Removed:** 0
- **Summary:** Edited TLCSDIS-1221.md, .gitignore

---

## Conversation

### 🧑 User

<manually_attached_skills>
The user has manually attached the following skills to their message.
These skills contain specific instructions or workflows that the user wants you to follow for this request.
Only read the files if needed, the full skill content is inlined here.

Skill Name: csdi-jira-issue-create
Path: /Users/a964286/.cursor/skills/csdi-jira-issue-create/SKILL.md
SKILL.md content:
# Jira Issue Create

## Trigger

User wants to create a single issue — a bug, ad-hoc task, or standalone story.

## Auth Handling

**REQUIRED RULE:** Follow `csdi-jira-auth.mdc` authentication procedures. Use `cloudId` from session context or `.cursor/jira-config.json`.

## Confirmation Gate

Story/Epic creation and Story terminal transitions require user confirmation (enforced by `beforeMCPExecution` hook). Present full draft before proceeding.

## Workflow

### 1. Authenticate

Verify MCP connection per Auth Handling above.

### 2. Discover Jira Context (once per invocation)

**Sprint discovery:** Run JQL `sprint in openSprints() AND project = TLCSDIS`, extract active sprint ID from `customfield_10020` where `state == "active"`, and reuse it for create calls in this workflow. If no active sprint: create in backlog and inform the user.

**Current user:**
- Call atlassianUserInfo to get accountId for assignee

### 3. Gather Context from Conversation

- **ON-DEMAND REFERENCE:** Load `csdi-jira-conventions-reference.md` for type definitions, taxonomy, and approved components.
- What is the issue about?
- Determine issue type (definitions per `csdi-jira-conventions.mdc`):
  - Error, defect, crash, regression → Bug
  - User story ("As a [user], I want [goal] so that [reason]") → Story
  - Specific work under a Story or Feature (often technical, assignable) → Task
  - Smaller part of a Task (granular tracking) → Sub-task

### 4. Discover Parent — List Candidates and Let User Choose

- For Story/Bug: JQL `issuetype = Epic AND project = TLCSDIS AND status != Done ORDER BY updated DESC` (maxResults=10)
- For Task/Sub-task: JQL `issuetype = Story AND sprint in openSprints() AND project = TLCSDIS AND status != Done ORDER BY updated DESC` (maxResults=10)
- Present list: "Which parent should this [Type] go under? 1. [KEY] — [Summary] ... 0. No parent (create as standalone)"
- Wait for user selection. If 0: create without parent, inform user

### 5. Select Description Template

- Bug: Use Bug template from templates.md (Impact, What Happened, Expected Behavior, Steps)
- Story: Use Story template (Objective, Acceptance Criteria, Background)
- Task: Use Task template (What Needs to Be Done, Definition of Done, Context)

### 6. Validate Labels

- Never invent ad-hoc labels
- If desired label doesn't match taxonomy, inform user and suggest closest match
- Apply multiple labels when work spans stakeholders or data sources

### 7. Confirmation Gate (if creating Story or Epic)

Present draft: "I'd like to create the following [Type] under [Parent KEY]. Here's what I plan to write: [draft]. Should I proceed? (yes / no / edit)"

### 8. Create Issue

- createJiraIssue with all fields populated
- Set parent to user's selection (or omit if standalone)
- Set sprint via additional_fields: { "customfield_10020": { "id": <sprint_id> } }
- Set assignee to current user
- Set priority "Medium" unless user specified otherwise

### 9. Verify (before reporting)

- Call getJiraIssue with the created issue key
- Confirm: sprint matches discovered sprint, parent matches user's selection (or is absent if standalone), labels and fields match draft
- ONLY report success after verification passes
- If verification fails: report the discrepancy and ask user how to proceed

### 10. Record Activity

If the issue was successfully created:
1. If creating `jira_progress/` for the first time, run the **Gitignore Guard** (see `skills/csdi-jira-progress-log/SKILL.md`) to ensure the directory is excluded from version control
2. Read `jira_progress/{PARENT_KEY}.md` if parent exists -- use parent context to enrich the new issue's description
3. Write a `[milestone]` entry to `jira_progress/{NEW_ISSUE_KEY}.md`:
   - **Did**: Created {issueType} "{summary}" under {parentKey}
   - **Blockers**: None

Follow the Activity Journal format in `skills/csdi-jira-progress-log/references/activity-journal-format.md`.

### 11. Report

"Created [TEAM-XXX] ([Type]) under [Parent KEY] — [Summary]. Assigned to [Sprint Name]. Priority: Medium."
</manually_attached_skills>
/csdi-jira-issue-create create story for legacy model training @dev/legacy/legacy_full_training.ipynb this is in progress, assign to me, to sprint 14. add technical details with csdi template

---

### 🤖 Assistant

I'll start by reading the necessary context files and the notebook to understand the work, then proceed with creating the Jira story.

---

### 🤖 Assistant

Let me find the templates reference and read more of the notebook, then get the Jira context.

---

### 🤖 Assistant

Now let me discover the Epic candidates and find the active sprint.

---

### 🤖 Assistant

The open sprints query returned empty issues — let me search for Sprint 14 directly to find the sprint ID.

---

### 🤖 Assistant

Found Sprint 14 (ID: 24894, "SDoH Sprint 14"). The most natural parent Epic for a legacy model training story is **TLCSDIS-1128 — "TE model formal retraining"** (In Progress). 

Here's my draft for your approval:

---

**Story Draft**

| Field | Value |
|---|---|
| **Type** | Story |
| **Summary** | Legacy Transformer baseline training with corrected pipeline and metrics parity |
| **Parent Epic** | TLCSDIS-1128 — TE model formal retraining |
| **Sprint** | SDoH Sprint 14 (ID: 24894) |
| **Assignee** | Xing, Daniel |
| **Priority** | Medium |
| **Status** | In Progress (will transition after creation) |

**Description:**

> **Objective**
> Establish a faithful legacy transformer baseline on the full 15M-member dataset using the corrected training pipeline, enabling direct performance comparison with the MoE experimental architecture (moe_flashattn_4).
>
> **Acceptance Criteria**
> - Legacy transformer trained on full dataset (`a834793_Combined_All_LOB_o3_train_ending`) with all three bug fixes applied (no log_softmax, corrected gradient clipping order, no double weight update)
> - Metrics tracked at full parity with moe_flashattn_4 framework: batch-level recall@K, precision@K, micro_recall@K, NDCG@20, positive_brier; epoch-level LossTracker statistics; gradient tier analysis; StreamingMetrics for validation
> - Early stopping integrated with configurable primary metric (micro_recall@20 default), sub-epoch validation at 500-step intervals, warmup period of 1000 steps
> - Results logged to `logs/{experiment_round}/legacy_replication/` in JSON format compatible with experimental comparison tooling
> - LOB embedding included in input tensor (`[age, gender, lob, codes]`) matching moe_flashattn_4 data format
>
> **Background**
> The legacy transformer serves as the production baseline for the TE retraining initiative. Three critical bugs were identified and corrected from the original `min_transformer_train.py`: redundant log_softmax (conflicting with BCEWithLogitsLoss), gradient clipping applied after optimizer step, and a double weight update. This story tracks the full-scale training run with these corrections and comprehensive metrics instrumentation to enable fair comparison against the MoE architecture.
>
> **Technical Details**
> - Architecture: Transformer encoder, 6 layers, 16 heads, 256-dim embedding, 512 FFN hidden, 0.05 dropout
> - Data: 200-day sequences, 80 codes/day, 75,516 input vocab, 6,297 target vocab
> - Optimizer: SGD (lr=0.01, momentum=0.9), CosineAnnealingLR, gradient clip 0.25
> - Training: Micro-batch 32 x 16 accumulation = 512 effective batch, 4 GPUs (DataParallel)
> - Loss: BCEWithLogitsLoss (no pos_weight)
> - Notebook: `dev/legacy/legacy_full_training.ipynb`

Should I proceed with creating this story?

---

### 🤖 Assistant

Now I see the correct parameter names. The field is `issueTypeName` not `issueType`, and `assignee_account_id` not `assignee`.

---

### 🤖 Assistant

Story created. Now let me verify the issue and transition it to "In Progress" as requested.

---

### 🤖 Assistant

Verification passed -- all fields match. Now transitioning to "In Progress" (transition ID 31) and adding a comment.

---

### 🤖 Assistant

Now let me record the activity in the progress journal.

---

### 🤖 Assistant

Need to create the `jira_progress/` directory and add it to `.gitignore` (Gitignore Guard per the skill instructions), then write the activity log.

---

### 🤖 Assistant

All done. Here's the summary:

**Created [TLCSDIS-1221](https://cvsdigital.atlassian.net/browse/TLCSDIS-1221)** (Story) under **TLCSDIS-1128** (TE model formal retraining)

| Field | Value |
|---|---|
| **Summary** | Legacy Transformer baseline training with corrected pipeline and metrics parity |
| **Sprint** | SDoH Sprint 14 |
| **Status** | In Progress |
| **Assignee** | Xing, Daniel |
| **Priority** | Medium |

**Verification passed:**
- Sprint: SDoH Sprint 14 (24894) confirmed
- Parent: TLCSDIS-1128 confirmed
- Assignee: Xing, Daniel confirmed
- Transitioned from Pending Approval to In Progress
- Status comment added with structured format
- Activity journal created at `jira_progress/TLCSDIS-1221.md`
- Gitignore guard applied (`jira_progress/` added to `.gitignore`)

