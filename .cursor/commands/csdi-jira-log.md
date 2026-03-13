You are logging development progress for a Jira issue. Follow the jira-progress-log skill workflow. Use the jira-conventions rule for project configuration.

The developer's input (issue key, description, or context): {user's input after the command}

## Workflow

1. Follow the jira-progress-log skill to resolve the target issue
2. Gather development context from git, conversation, and recent files
3. Determine entry type ([development], [decision], [milestone], [blocker])
4. Synthesize and confirm the entry draft with the user
5. Write the entry to jira_progress/{ISSUEKEY}.md
6. Report what was logged
