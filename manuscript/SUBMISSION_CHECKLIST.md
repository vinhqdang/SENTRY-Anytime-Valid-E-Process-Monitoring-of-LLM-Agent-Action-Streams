# IEEE TDSC Submission Checklist — SENTRY

**Target:** IEEE Transactions on Dependable and Secure Computing,
Special Issue *"Safety, Alignment, and Responsibility of Large Language Models."*
**Article type:** SI — Safety, Alignment, and Responsibility of Large Language Models
**Deadline:** 31 December 2026
**Portal:** IEEE Author Portal (Research Exchange)

---

## What to upload, and to which slot

| Portal slot | File to upload | Notes |
|---|---|---|
| **Main Manuscript** (required) | `sentry_TDSC_main.zip` | LaTeX bundle: `sentry_ieee.tex`, `sentry_ieee.bbl`, `references_ieee.bib`, `figures/`, `IEEEtran.cls`, `IEEEtran.bst`. Contains the manuscript ONLY — no supplementary material, no cover letter (per portal instructions). 12 pages. |
| **Supplementary Material for Review** (optional) | `appendix.pdf` | Reproducibility + hyperparameters, extended signal definitions, synthetic-validation figure, SR-vs-CUSUM. Published as supplementary material. |
| **LaTeX Supplementary File** (optional) | `appendix.tex` | Source for the supplementary PDF (upload if the portal wants the source). |
| **Cover letter / Comments** (optional) | `cover_letter.pdf` | Not shown to reviewers, not published. |
| Main Document – Tracked Changes | — | N/A for a first submission (revisions only). |
| Previously Published | — | N/A. (The earlier AAMAS desk-reject is not a publication; nothing to declare.) |
| Image | — | Not needed — figures are inside the main bundle. |

## Portal metadata to enter by hand
- **Title:** SENTRY: Anytime-Valid E-Process Monitoring of LLM Agent Action Streams
- **Author:** Quang-Vinh Dang — British University Vietnam, Hung Yen, Vietnam — vinh.dq4@buv.edu.vn
- **Keywords:** large language model agents; prompt injection; e-values; sequential change detection; anytime-valid inference; runtime monitoring
- Confirm sole authorship; no conflicts of interest; no funding.

## Verification done
- Main bundle compiles standalone (`pdflatex → bibtex → pdflatex ×2`): **12 pages, 0 errors, 0 undefined references.**
- Bibliography: 15 peer-reviewed + 14 arXiv entries; DOIs/URLs verified.

## For camera-ready (only if accepted — do NOT add now)
Author biography (kept out of the 12-page initial submission; the block is
commented at the end of `sentry_ieee.tex`, ready to uncomment):

> Quang-Vinh Dang is with British University Vietnam, Hung Yen, Vietnam. His
> research interests include machine learning, anytime-valid and sequential
> statistical inference, and the safety and security of large-language-model-based
> agents.

*(Please expand with your degrees/positions before camera-ready.)*

## Housekeeping
- Rotate the OpenRouter API key used for data collection (it appeared in chat logs).
