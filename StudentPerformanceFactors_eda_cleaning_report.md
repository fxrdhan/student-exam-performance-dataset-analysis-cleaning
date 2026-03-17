# Student Performance Factors: EDA and Cleaning Report

## Dataset Overview
- Source file: `StudentPerformanceFactors.csv`
- Original shape: `6607 rows x 20 columns`
- Clean shape: `6607 rows x 20 columns`
- Exact duplicate rows before cleaning: `0`
- Exact duplicate rows after cleaning: `0`

## Cleaning Decisions
- Standardized text labels by trimming whitespace.
- Converted blank categorical values to missing before imputation.
- Filled missing values in `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` with `Unknown`.
- Capped `Exam_Score` to the valid domain `[0, 100]`; only one row required correction (`101 -> 100`).
- Did not delete statistical outliers identified by IQR because those rows remain domain-plausible and may carry real signal for tree-based models.

## Missing Values Before Cleaning
```text
                            missing_count  missing_pct
Hours_Studied                           0         0.00
Attendance                              0         0.00
Parental_Involvement                    0         0.00
Access_to_Resources                     0         0.00
Extracurricular_Activities              0         0.00
Sleep_Hours                             0         0.00
Previous_Scores                         0         0.00
Motivation_Level                        0         0.00
Internet_Access                         0         0.00
Tutoring_Sessions                       0         0.00
Family_Income                           0         0.00
Teacher_Quality                        78         1.18
School_Type                             0         0.00
Peer_Influence                          0         0.00
Physical_Activity                       0         0.00
Learning_Disabilities                   0         0.00
Parental_Education_Level               90         1.36
Distance_from_Home                     67         1.01
Gender                                  0         0.00
Exam_Score                              0         0.00
```

## Missing Values After Cleaning
```text
                            missing_count  missing_pct
Hours_Studied                           0          0.0
Attendance                              0          0.0
Parental_Involvement                    0          0.0
Access_to_Resources                     0          0.0
Extracurricular_Activities              0          0.0
Sleep_Hours                             0          0.0
Previous_Scores                         0          0.0
Motivation_Level                        0          0.0
Internet_Access                         0          0.0
Tutoring_Sessions                       0          0.0
Family_Income                           0          0.0
Teacher_Quality                         0          0.0
School_Type                             0          0.0
Peer_Influence                          0          0.0
Physical_Activity                       0          0.0
Learning_Disabilities                   0          0.0
Parental_Education_Level                0          0.0
Distance_from_Home                      0          0.0
Gender                                  0          0.0
Exam_Score                              0          0.0
```

## Change Log Summary
```text
                          rule                   column  rows_changed
categorical_missing_to_unknown       Distance_from_Home            67
categorical_missing_to_unknown Parental_Education_Level            90
categorical_missing_to_unknown          Teacher_Quality            78
                clip_to_domain               Exam_Score             1
```

## Numeric Summary After Cleaning
```text
                    count   mean    std   min   25%   50%   75%    max
Hours_Studied      6607.0  19.98   5.99   1.0  16.0  20.0  24.0   44.0
Attendance         6607.0  79.98  11.55  60.0  70.0  80.0  90.0  100.0
Sleep_Hours        6607.0   7.03   1.47   4.0   6.0   7.0   8.0   10.0
Previous_Scores    6607.0  75.07  14.40  50.0  63.0  75.0  88.0  100.0
Tutoring_Sessions  6607.0   1.49   1.23   0.0   1.0   1.0   2.0    8.0
Physical_Activity  6607.0   2.97   1.03   0.0   2.0   3.0   4.0    6.0
Exam_Score         6607.0  67.24   3.89  55.0  65.0  67.0  69.0  100.0
```

## Correlation With Exam_Score
```text
Attendance           0.581
Hours_Studied        0.446
Previous_Scores      0.175
Tutoring_Sessions    0.156
Physical_Activity    0.028
Sleep_Hours         -0.017
```

## Categorical Distributions After Cleaning
```text
[Parental_Involvement]
Parental_Involvement
Medium    3362
High      1908
Low       1337

[Access_to_Resources]
Access_to_Resources
Medium    3319
High      1975
Low       1313

[Extracurricular_Activities]
Extracurricular_Activities
Yes    3938
No     2669

[Motivation_Level]
Motivation_Level
Medium    3351
Low       1937
High      1319

[Internet_Access]
Internet_Access
Yes    6108
No      499

[Family_Income]
Family_Income
Low       2672
Medium    2666
High      1269

[Teacher_Quality]
Teacher_Quality
Medium     3925
High       1947
Low         657
Unknown      78

[School_Type]
School_Type
Public     4598
Private    2009

[Peer_Influence]
Peer_Influence
Positive    2638
Neutral     2592
Negative    1377

[Learning_Disabilities]
Learning_Disabilities
No     5912
Yes     695

[Parental_Education_Level]
Parental_Education_Level
High School     3223
College         1989
Postgraduate    1305
Unknown           90

[Distance_from_Home]
Distance_from_Home
Near        3884
Moderate    1998
Far          658
Unknown       67

[Gender]
Gender
Male      3814
Female    2793
```

## IQR Outlier Scan
```text
           column  lower_bound  upper_bound  outlier_count
    Hours_Studied          4.0         36.0             43
       Attendance         40.0        120.0              0
      Sleep_Hours          3.0         11.0              0
  Previous_Scores         25.5        125.5              0
Tutoring_Sessions         -0.5          3.5            430
Physical_Activity         -1.0          7.0              0
       Exam_Score         59.0         75.0            104
```

## Notes for Decision Trees
- Using `Unknown` keeps missingness explicit so a decision tree can branch on it.
- No rows were removed, so sample size and class distribution remain intact.
- Only domain-invalid values were corrected; statistical extremes were retained.