# Publication validation

Checked on 17 November 2026 using Python 3.11 and the project’s pinned numerical packages.

- All six project unit tests passed.
- The standalone reproduction completed on the checksum-verified Hillstrom dataset.
- Primary contrast, comparator results, policy values, budget curves and quintile summaries matched the saved results within relative tolerance 1e-9 and absolute tolerance 1e-12.
- The reproduced primary estimate was 5.392886071264886, with interval 0.544764306511066 to 9.734564166234685, on 21,307 evaluation rows.
- The published code and analysis plans are byte-identical to the reviewed project source. The original evaluation lock records an earlier pre-formatting source state. Formatting changed source hashes after the original run, so those historical hashes are not presented as current-source hashes. Current file hashes are recorded in SHA256SUMS.json.
- Raw data, fitted models, virtual environments and local reproduction folders are excluded from this publication package. Data retrieval remains checksum verified.

The original evaluation files are preserved, not overwritten. Packaging adds a standalone runner and documentation without changing the modelling code. This validation does not remove the stated limits of an offline historical evaluation.
