You are marking development on an issue as complete. This does NOT mean "Done" — it means "Dev Complete" for Stories or "Verifying" for Sub-tasks. Follow the jira-progress-sync skill workflow. Use the jira-conventions rule for project configuration and jira-language rule for content translation.

The developer's input (issue key or description): {user's input after the command}

## MANDATORY: Issue-Type-Aware Status Resolution

<HARD-RULE>
"Marking done" does NOT mean transition to status "Done." It means the development phase is complete.
The target status depends on the issue type:

| Issue Type | Target Status | Transition ID | NEVER use |
|---|---|---|---|
| **Story** | Dev Complete | 41 | Do NOT use "Accepted" (id=61) or "Done" |
| **Feature** | Dev Complete | 41 | Do NOT use "Accepted" (id=61) or "Done" |
| **Sub-task** | Verifying | 51 | Do NOT use "Done" (id=31) — that is the TERMINAL status |

"Done" (id=31) for Sub-tasks is equivalent to "Accepted" (id=61) for Stories — it means fully closed.
"Verifying" (id=51) for Sub-tasks is equivalent to "Dev Complete" (id=41) for Stories.
</HARD-RULE>

## Workflow

1. Identify the specific issue (by key or description match via getJiraIssue)
2. Read the `issuetype.name` field from the response — this determines the target status from the table above
3. Call getTransitionsForJiraIssue to get available transitions
4. Match the correct target status NAME from the table above (NOT "Done" for Sub-tasks)
5. Add a progress comment in business language
6. Call transitionJiraIssue with the matched transition ID
7. For Story transitions to "Accepted" only: respect the Confirmation Gate

## Example: Sub-task Completed

If issue TLCSDIS-1053 is a Sub-task:
- getTransitionsForJiraIssue returns: Not Started (11), In Progress (21), Done (31), Cancel (41), Verifying (51)
- Correct: transition id = "51" (Verifying) ← THIS ONE
- WRONG: transition id = "31" (Done) ← NEVER use this for "development complete"
