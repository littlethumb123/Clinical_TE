# Pre-Experiment Registration

Generate a structured pre-registration document before any ML experiment.

## Instructions

Ask the user for (or infer from context):
1. Experiment name
2. What they're testing (the hypothesis)
3. What they expect to see

Then generate and save this document:

## Template

```markdown
## Pre-Registration: [Experiment Name]
**Date**: [today]
**Hypothesis**: [What the user thinks will happen and why]
**Expected Result**: [Specific metric values or behaviors expected]
**Refutation Criterion**: If I see [specific result], my hypothesis is wrong
**Scientific Variable**: [The ONE thing being changed]
**Nuisance Variables**: [Things being tuned/optimized over]
**Fixed Variables**: [Things held constant]
**Controls**: [What stays the same vs baseline]
**Baseline Reference**: [What existing result this compares against]
**Diagnostics to Instrument**: [Gradient tracking, logit histograms, etc.]
**Estimated Cost**: [GPU-hours, wall time, $]
**Method Maturity**: [Are all methods production-proven?]
```

Save to: `docs/experiments/YYYY-MM-DD-[name]-preregistration.md`
