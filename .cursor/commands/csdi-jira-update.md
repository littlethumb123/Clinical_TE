You are syncing the developer's progress to Jira. Follow the jira-progress-sync skill workflow. Use the jira-conventions rule for project configuration and jira-language rule for content translation. Confirmation is hook-enforced; present full draft before stakeholder-visible writes.

The developer's update: {user's input after the command}

<HARD-RULE>
Stories and Sub-tasks have DIFFERENT status workflows. When transitioning:
- Story/Feature completed → "Dev Complete" (id=41). NOT "Accepted."
- Sub-task completed → "Verifying" (id=51). NOT "Done" (id=31).
- "Done" for Sub-tasks = fully closed/accepted. "Verifying" = development complete.
</HARD-RULE>

Execute the full progress sync workflow:
1. Identify the relevant issue(s) and read their issue type from getJiraIssue
2. Determine what changed and resolve the correct target status per issue type using the table above
3. Call getTransitionsForJiraIssue and match the correct target status NAME
4. Translate to business language, add a comment, and transition status
5. Always respect the Confirmation Gate for terminal transitions (Story → "Accepted", Sub-task → "Done")
