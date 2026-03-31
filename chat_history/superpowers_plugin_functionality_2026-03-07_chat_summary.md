# Superpowers plugin functionality

## Metadata

- **Conversation ID:** b932d3e8-e6e3-46c8-b661-6e9a83eebf76
- **Mode:** agent
- **Created:** 2026-03-07 15:07:15
- **Last Updated:** 2026-03-07 15:07:15
- **Branch:** main
- **Files Changed:** 0
- **Lines Added:** 0
- **Lines Removed:** 0
- **Summary:** Here's an overview of the Superpowers plugin — a core skills…

---

## Conversation

### 🧑 User

Show me how the Superpowers plugin works

---

### 🤖 Assistant

Here's an overview of the **Superpowers** plugin — a core skills library that enhances your development workflow with structured patterns for TDD, debugging, collaboration, and more.

---

## Slash Commands

These can be invoked directly with `/` in chat:

- **`/brainstorm`** — Explores requirements, intent, and design *before* implementation. Use this before any creative work like creating features, building components, or modifying behavior.
- **`/write-plan`** — Creates a detailed implementation plan with bite-sized tasks from a spec or set of requirements.
- **`/execute-plan`** — Executes a written plan in batches with review checkpoints, so you can validate progress at each stage.

---

## Skills (auto-invoked or manually with `/`)

These are specialized workflows the agent uses automatically when appropriate, or you can invoke them directly:

| Skill | When It Activates |
|---|---|
| **brainstorming** | Before any creative work — features, components, behavior changes |
| **test-driven-development** | Before writing implementation code for any feature or bugfix |
| **systematic-debugging** | When encountering any bug, test failure, or unexpected behavior |
| **writing-plans** | When you have a spec/requirements for a multi-step task |
| **executing-plans** | When you have a written plan to execute with review checkpoints |
| **dispatching-parallel-agents** | When facing 2+ independent tasks with no shared state |
| **subagent-driven-development** | When executing plans with independent tasks in the current session |
| **requesting-code-review** | After completing tasks or major features, before merging |
| **receiving-code-review** | When handling code review feedback — ensures rigor over blind agreement |
| **verification-before-completion** | Before claiming work is done — requires evidence before assertions |
| **finishing-a-development-branch** | When implementation is complete and you need to integrate (merge, PR, cleanup) |
| **using-git-worktrees** | When starting feature work that needs isolation from the current workspace |
| **writing-skills** | When creating, editing, or verifying skills |

---

## Subagents

- **`code-reviewer`** — Automatically invoked (or manually triggered) after a major project step is completed. Reviews the implementation against the original plan and coding standards.

---

## Hooks (automatic)

- **`SessionStart`** — Runs automatically at the beginning of each session to establish context and available skills.

---

## Key Workflow Patterns

1. **Plan-first development**: `/write-plan` to design, then `/execute-plan` to implement in reviewed batches.
2. **TDD loop**: The TDD skill activates before implementation, guiding you to write tests first.
3. **Debug rigorously**: The systematic-debugging skill kicks in on failures, preventing guesswork fixes.
4. **Verify before claiming done**: The verification skill ensures commands are actually run and output confirmed before any "it works" claims.

The plugin essentially enforces disciplined engineering practices — plan before you code, test before you implement, verify before you ship.

The user wants an overview of the Superpowers plugin they just installed. Let me provide a clear summary based on the system reminder information.

