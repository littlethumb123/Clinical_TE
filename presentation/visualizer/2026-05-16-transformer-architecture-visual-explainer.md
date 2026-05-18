# Clinical TE Transformer Visual Explainer Implementation Plan

> **For Copilot or other skill-enabled agents:** REQUIRED SUB-SKILL: Use `/executing-plans` to implement this plan task-by-task.

**Goal:** Generate a self-contained HTML explainer that visualizes the Clinical TE architecture, data flow, and stage-by-stage tensor transformations for one member history.

**Architecture:** Build a single HTML page in `presentation/visualizer/` with a Mermaid overview, CSS cards for the transformation strip, deterministic JavaScript heatmaps for the tensor visuals, and a dedicated temporal-encoder walkthrough that shows what the same causal mask achieves across layers for a selected day. Source the content from the existing presentation transcript and data-transformation guide so the page stays presentation-aligned.

**Tech Stack:** HTML, CSS, vanilla JavaScript, Mermaid.js via CDN

---

### Task 1: Lock The Narrative And Output Paths

**Files:**
- Create: `presentation/visualizer/2026-05-16-transformer-architecture-visual-explainer-design.md`
- Create: `presentation/visualizer/2026-05-16-transformer-architecture-visual-explainer.md`

**Step 1: Save the design doc**

Write the approved hybrid design into the design doc with the target sections, key messages, and output file path.

**Step 2: Save this implementation plan**

Create a short execution plan that points to the exact HTML output file and validation steps.

**Step 3: Review the plan text for path correctness**

Confirm the output path points to `presentation/visualizer/clinical-te-transformer-architecture-explainer.html`.

### Task 2: Build The Visual Explainer Page

**Files:**
- Create: `presentation/visualizer/clinical-te-transformer-architecture-explainer.html`

**Step 1: Create the page shell**

Add the HTML document, typography, responsive layout, section navigation, and hero summary.

**Step 2: Add the architecture overview**

Add a Mermaid flowchart showing the full path from member history through next-day prediction, including demographic injection and the six-layer temporal encoder.

**Step 3: Add the transformation strip**

Add six stage cards that show one member's data moving through the model:

- Raw day codes
- Embedding lookup `[80, 256]`
- Learned attention pooling `[1, 256]`
- Stacked days `[200, 256]`
- Temporal encoder output `[200, 256]`
- Final member embedding `[256]`

**Step 4: Add the temporal reasoning section**

Add the causal mask visual, the internals of one layer, and a selected-day walkthrough that explains the transcript's `Layer 0 -> Layer 1 -> Layer 5` story.

**Step 5: Add the shape ledger**

Add a table that maps every major transformation to its tensor shape and interpretation.

### Task 3: Add Lightweight Interaction

**Files:**
- Modify: `presentation/visualizer/clinical-te-transformer-architecture-explainer.html`

**Step 1: Render the Mermaid diagram**

Initialize Mermaid with the page palette and make sure the overview diagram is readable.

**Step 2: Populate deterministic heatmaps**

Use small deterministic JavaScript helpers to generate consistent schematic tensor visuals without requiring model checkpoint access.

**Step 3: Add scroll spy and zoom controls**

Add sticky section navigation and zoom controls for the Mermaid container.

### Task 4: Validate The Output

**Files:**
- Test: `presentation/visualizer/clinical-te-transformer-architecture-explainer.html`

**Step 1: Run HTML validation**

Run: `xmllint --html --noout /Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/presentation/visualizer/clinical-te-transformer-architecture-explainer.html`

Expected: No fatal parse errors.

**Step 2: Open the page in the browser**

Run: `open /Users/a964286/Documents/Projects/Clinical_TE/Clinical_TE/presentation/visualizer/clinical-te-transformer-architecture-explainer.html`

Expected: The page loads, the Mermaid diagram renders, and the stage visuals appear.

**Step 3: Confirm the narrative matches the architecture transcript**

Check that the page covers the same core ideas: hierarchical input, learned attention pooling, demographic residual injection, causal temporal modeling, and final embedding extraction.

**Step 4: Commit if requested later**

If the user later asks for versioning, commit the design and plan documents together with a single descriptive message.