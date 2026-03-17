# Student Exam Performance Dataset Analysis

This repository contains the raw dataset, the cleaned dataset, the cleaning audit log, and the Python script used to reproduce the cleaning process.

## Dataset Source

Source page:
https://www.kaggle.com/datasets/grandmaster07/student-exam-performance-dataset-analysis/data

## Repository Contents

- `StudentPerformanceFactors.csv`: raw dataset
- `StudentPerformanceFactors_clean.csv`: cleaned dataset ready for analysis or modeling
- `StudentPerformanceFactors_cleaning_changes.csv`: cell-level audit trail of all cleaning changes
- `StudentPerformanceFactors_eda_cleaning_report.md`: EDA summary and cleaning rationale
- `eda_clean_student_performance.py`: reproducible cleaning pipeline

## Target Column

For supervised modeling, the target column is `Exam_Score`.

## Reproduce the Clean Dataset

```bash
python eda_clean_student_performance.py
```
