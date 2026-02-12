# Input formats (data-free repository)

This repository does **not** ship any empirical data. You provide your own inputs.

## 1) Planning artifact (CoRe-like table) as *wide CSV*

The pipeline expects a CSV file with:

- Row 1: header row  
  - Column 1 is empty (or a label like `GuidingQuestion`)
  - Columns 2..K are column labels (e.g., "Big Ideas")
- Column 1: row labels (e.g., guiding questions)
- Remaining cells: the text entries (may be empty)

Example (schematic):

|           | BigIdea1 | BigIdea2 |
|-----------|----------|----------|
| Q1        | ...      | ...      |
| Q2        | ...      |          |

Save as UTF-8 CSV.

## 2) Reflection as plain text

A `.txt` file containing the full written reflection. The pipeline will segment it into
sentences using punctuation-based heuristics.
