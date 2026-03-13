# Post-Experiment Review

Generate a structured post-mortem document after any ML experiment.

## Instructions

Ask the user for (or infer from context):
1. Which experiment (link to pre-registration if exists)
2. What happened
3. Whether it matched expectations

Then generate and save this document:

## Template

```markdown
## Post-Mortem: [Experiment Name]
**Date**: [today]
**Pre-Registration Reference**: [link to pre-reg doc, if exists]

### Expected vs Actual
| Metric | Expected | Actual | Delta |
|--------|----------|--------|-------|
| [metric1] | [value] | [value] | [+/-] |

### Hypothesis Status
[Confirmed / Rejected / Refined to: ___]

### Surprise
What I didn't expect: ___

### Root Cause Analysis
Why the result occurred: ___

### Lesson Learned
What I now know that I didn't before: ___

### Knowledge Update
- Updated understanding: ___
- Revised beliefs about: ___

### Next Experiment
Based on this result, the next experiment should: ___
Hypothesis for next: ___
```

Save to: `docs/experiments/YYYY-MM-DD-[name]-postmortem.md`
