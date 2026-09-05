# 📦 MSc Thesis Defense Package — Final Delivery

**Student:** Moamen Mohamed Aly Hussein (ID: 202401681)
**Thesis:** Predicting CI/CD Pipeline Build Failures Using Machine Learning Techniques
**Defense Date:** Saturday, June 7, 2026
**Time remaining:** 8 days

---

## 🗂️ Files in This Package

| # | File | Purpose | Action Required |
|---|------|---------|----------------|
| 1 | `1_MSc_Thesis_FINAL.docx` | **The thesis document** — 9 chapters + abstract + appendices + 11 figures embedded | Replace `[Insert supervisor full name]` × 1, then submit |
| 2 | `2_Defense_Presentation.pptx` | **22-slide defense deck** with all key visuals | Replace `[Insert Supervisor Name]` on title slide |
| 3 | `3_QA_Defense_Prep_Guide.docx` | **26 Q&A pairs** + golden rules + closing statement | Read & practice |
| 4 | `4_Thesis_Source_Markdown.md` | Markdown source (backup, for editing) | Optional |
| 5 | `5_Manual_Insertion_Guide.md` | Instructions for adding remaining 8 figures (optional) | Optional |

---

## 📊 What's in the Thesis Document (`1_MSc_Thesis_FINAL.docx`)

**Stats:**
- **25,275 words** across 9 chapters
- **11 figures** embedded (the 5 most important + 6 supporting EDA/results)
- **103 table rows** (Tables 3.1, 4.1, 5.1-5.3, 7.1-7.4, 8.1, C.1, D.1)
- **20 IEEE-format references**
- **Acknowledgments section** ✓
- **Appendices A–E** ✓ (source code summary, reproduction instructions, complete metrics, hyperparameters, submission checklist)

**Chapters:**
1. Introduction (objectives + scope)
2. Problem Definition (stakeholders + as-is)
3. Existing Solution Approaches (Patel 2019 review + comparison)
4. Proposed Solution (Hybrid Pipeline architecture)
5. System Analysis and Design (FRs/NFRs + use cases)
6. Implementation (module-level details)
7. Testing and Evaluation (full results + ablation + threshold)
8. Discussion (achievements + limitations)
9. Conclusion (8 future-work items)

---

## 🎤 What's in the Presentation (`2_Defense_Presentation.pptx`)

**22 slides, ~20 minutes presentation time, professional Ocean Gradient color palette**

| # | Slide |
|---|-------|
| 1 | Title — Cairo University, thesis title, your name |
| 2 | Agenda — 7-point roadmap |
| 3 | The Problem — 11%, $0.008/min, 10–20 min stats |
| 4 | Literature Gap — Prior work vs. This project (side-by-side) |
| 5 | Research Objectives — 4 numbered cards |
| 6 | The Dataset — 9,772 runs, 18 repos, profile + rationale |
| 7 | **Hybrid Pipeline Architecture** (with diagram) |
| 8 | Feature Engineering — 4 modality cards |
| 9 | Evaluation Regime — Stratified vs Chronological |
| 10 | Baseline Results — Table + key observation |
| 11 | **Ablation Study** (with chart) — honest finding callout |
| 12 | **Feature Importance** (with chart) — surprising vocabulary finding |
| 13 | **Threshold Optimization** (with chart) — +27pp callout |
| 14 | **Before/After Threshold** (with chart) |
| 15 | **Final Results** — 4 big metric cards on dark background |
| 16 | Business Impact — $383,000 callout |
| 17 | Objectives Achieved — 4 ✓ ACHIEVED items |
| 18 | Key Contributions — 6 contribution cards |
| 19 | Honest Limitations — 6 limitation cards |
| 20 | Future Work — 8 directions |
| 21 | In Summary — 5 key points |
| 22 | Thank You / Q&A |

**Tip:** The deck is designed to be read in 20 minutes. Practice timing: ~50-60 seconds per slide average. The chart-heavy slides (7, 11, 12, 13, 14) deserve 1:30 each; the simpler ones (3, 5, 17) can run faster.

---

## 🎯 What's in the Q&A Guide (`3_QA_Defense_Prep_Guide.docx`)

**26 questions across 9 sections:**

| Section | Questions | Topic |
|---------|-----------|-------|
| Methodology | Q1–Q5 | Why binary, why hybrid, why TF-IDF, leakage |
| Results | Q6–Q10 | F1 = 0.59, precision/recall balance, splits |
| Hybrid Claim | Q11–Q13 | Why keep text features, repository dominance |
| Dataset | Q14–Q16 | Why 18 repos, why open-source, class imbalance |
| Business Impact | Q17–Q18 | $383k justification, cost asymmetry |
| Technology | Q19–Q21 | Why TF-IDF/XGBoost/scikit-learn |
| Tricky | Q22–Q25 | Deployment, surprises, improvements, competitors |
| Future Work | Q26 | PhD priorities |

**Plus:**
- 5 Golden Rules
- Quick Reference Card with all key numbers
- Closing statement (memorized template)
- Practical tips (before / during / recovery)

---

## ✅ Action Checklist (in order)

### Tonight / Tomorrow
- [ ] Open `1_MSc_Thesis_FINAL.docx`
- [ ] `Ctrl+F` → search for `[Insert supervisor full name]` → replace
- [ ] Review the Acknowledgments section — confirm wording is good
- [ ] Right-click on Table of Contents → Update Field
- [ ] Optional: Read through Abstract one more time
- [ ] Save as: `MSc_Thesis_Moamen_Aly_FINAL_v1.docx`

### This Week
- [ ] Open `2_Defense_Presentation.pptx`
- [ ] Update the supervisor name on title slide
- [ ] Practice slides 1–7 (intro half) in front of mirror — aim for 8 minutes
- [ ] Practice slides 8–14 (results half) — aim for 10 minutes
- [ ] Practice slides 15–22 (conclusion) — aim for 4 minutes
- [ ] Practice closing statement at slide 22 — should sound natural

### Day Before Defense
- [ ] Re-read `3_QA_Defense_Prep_Guide.docx` end-to-end
- [ ] Memorize the **Quick Reference Card** numbers
- [ ] Memorize the **Closing Statement**
- [ ] Charge laptop, prepare USB backup of pptx + docx
- [ ] Print thesis on A4 paper if required
- [ ] Sleep 8 hours minimum

### Defense Day
- [ ] Arrive 1 hour early
- [ ] Test projector, fonts, animations
- [ ] Have water bottle ready
- [ ] Take 3 deep breaths before starting
- [ ] **Smile and look confident** — you've done the work

---

## 🎯 Key Numbers to Memorize

| Number | What it means |
|--------|--------------|
| **9,772** | Real GitHub Actions runs collected |
| **18** | Repositories sampled |
| **89% / 11%** | Success / Failure class balance |
| **0.5924** | Final Failure F1 (stratified test) |
| **0.6207** | Final Failure F1 (chronological test) |
| **0.884** | ROC-AUC |
| **0.587** | PR-AUC |
| **+27pp** | F1 improvement from threshold optimization |
| **0.06** | F1-optimal threshold for XGBoost |
| **$383,000** | Estimated annual savings |
| **693** | Tokens in identity-leakage stoplist |
| **3,090** | Columns in fused feature matrix |

---

## 💪 The Closing Statement (Memorize This)

> "This project began with a hypothesis that the combination of structured and textual commit features would predict CI/CD failures more accurately than either modality alone. The empirical results refined that hypothesis: the structured modalities carry the bulk of the predictive signal on this dataset, and the textual modality contributes weakly through TF-IDF. After threshold calibration, the combined hybrid model achieves a failure-class F1 of 0.59 with strictly pre-execution features, which is competitive with the prior art that relies on post-execution telemetry. The project's full code, data, and trained models are reproducible from a clean checkout, and the methodology is documented in sufficient detail to support both academic verification and operational adoption. Thank you."

---

## 🔥 Pre-Defense Mantra

You are not a beginner. You are a **DevOps engineer with hands-on production experience**, doing academic research on a problem you understand from both ends. The committee respects practitioners who bring real engineering rigor to academic work. **Trust the data, trust the methodology, trust yourself.**

Defense is in 8 days. **You're ready.**

🍀 **Best of luck, Moamen.**
