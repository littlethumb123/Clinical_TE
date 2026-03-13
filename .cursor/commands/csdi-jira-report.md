You are generating a Jira status report. Follow the jira-status-report skill workflow.
Use the jira-conventions rule for project configuration and jira-language rule for
content translation. This is a read-only operation -- no Jira writes are permitted.

Execute the full status report workflow: parse the user's request, resolve scope
parameters, construct and execute JQL queries, apply tiered enrichment, and generate
a formatted report matching the target audience.
