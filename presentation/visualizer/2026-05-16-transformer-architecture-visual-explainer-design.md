# Transformer Architecture Visual Explainer Design

## Goal

Generate a self-contained HTML explainer that shows both the Clinical TE transformer architecture and how one member history is transformed from padded clinical codes into a final 256-dimensional embedding.

## Audience

- Presentation audiences who need an intuitive architecture and data-flow story
- Technical reviewers who need tensor shapes, masking behavior, and stage semantics

## Chosen Approach

Use a hybrid visual page rather than a single static diagram.

1. A topology-focused architecture overview at the top shows the end-to-end system: code embedding lookup, learned attention pooling, demographic injection, temporal encoder, last-day embedding extraction, and next-day prediction.
2. A horizontal transformation strip shows what the data looks like for one member as it moves through the model: raw codes, code embeddings, pooled day vector, stacked day sequence, contextualized temporal sequence, and final member embedding.
3. A temporal reasoning section explains the causal mask, the internals of one layer, and a selected-day walkthrough showing what the same mask achieves across Layer 0, Layer 1, and Layer 5.
4. A shape ledger makes the tensor transformations explicit so the page can be used as both a stakeholder explainer and a technical reference.

## Content Sources

- `presentation/architecture_walkthrough_transcript.md` for component semantics, naming, and training/evaluation context
- `presentation/data_transformation_visualization_guide.md` for the transformation-strip concept and example member/day data

## Visual Direction

- Blueprint-editorial hybrid: deep blue, slate, and amber accents with subtle grid atmosphere
- Large serif headings for hierarchy, monospace labels for tensor shapes and operations
- Mermaid for the system overview, CSS cards and deterministic heatmaps for data-shape visuals

## Key Messages

- The model input is not a flat feature row. It starts as a padded hierarchy: up to 200 days, each day with up to 80 unordered codes.
- The daily encoder compresses a set of code embeddings into one day vector.
- Demographic embeddings condition each day vector before temporal modeling.
- The temporal encoder preserves causal integrity: day `t` can only depend on days `1..t`.
- The final 256-dimensional member embedding is the last valid day representation after six layers of contextualization.

## Validation Plan

- Validate the generated HTML for structural issues.
- Open the page in the browser and verify that the Mermaid diagram renders with non-zero size, the heatmaps populate, and the temporal walkthrough reads clearly on a narrow viewport.
- Confirm that the shape ledger matches the intended sequence of transformations.

## Output Path

- Generated explainer: `presentation/visualizer/clinical-te-transformer-architecture-explainer.html`

## Notes

- This design doc is saved alongside the explainer assets in `presentation/visualizer/` for traceability.
- No git commit is created here because that was not requested.