from pathlib import Path

import pandas as pd


INPUT_PATH = Path("StudentPerformanceFactors.csv")
OUTPUT_PATH = Path("StudentPerformanceFactors_clean.csv")
CHANGES_PATH = Path("StudentPerformanceFactors_cleaning_changes.csv")
REPORT_PATH = Path("StudentPerformanceFactors_eda_cleaning_report.md")

EXPECTED_CATEGORIES = {
    "Parental_Involvement": {"High", "Low", "Medium"},
    "Access_to_Resources": {"High", "Low", "Medium"},
    "Extracurricular_Activities": {"No", "Yes"},
    "Motivation_Level": {"High", "Low", "Medium"},
    "Internet_Access": {"No", "Yes"},
    "Family_Income": {"High", "Low", "Medium"},
    "Teacher_Quality": {"High", "Low", "Medium", "Unknown"},
    "School_Type": {"Private", "Public"},
    "Peer_Influence": {"Negative", "Neutral", "Positive"},
    "Learning_Disabilities": {"No", "Yes"},
    "Parental_Education_Level": {"College", "High School", "Postgraduate", "Unknown"},
    "Distance_from_Home": {"Far", "Moderate", "Near", "Unknown"},
    "Gender": {"Female", "Male"},
}

MISSING_IMPUTATION = {
    "Teacher_Quality": "Unknown",
    "Parental_Education_Level": "Unknown",
    "Distance_from_Home": "Unknown",
}

NUMERIC_DOMAIN_BOUNDS = {
    "Exam_Score": (0, 100),
}


def add_change(changes, row_number, column, old_value, new_value, rule, rationale):
    changes.append(
        {
            "row_number": row_number,
            "column": column,
            "old_value": old_value,
            "new_value": new_value,
            "rule": rule,
            "rationale": rationale,
        }
    )


def trim_strings(df, changes):
    object_columns = df.select_dtypes(include="object").columns
    for column in object_columns:
        original = df[column].copy()
        trimmed = df[column].astype("string").str.strip()
        trimmed = trimmed.replace({"": pd.NA})

        changed_mask = original.notna() & trimmed.notna() & (
            original.astype(str) != trimmed.astype(str)
        )
        for index in df.index[changed_mask]:
            add_change(
                changes=changes,
                row_number=index + 1,
                column=column,
                old_value=original.iloc[index],
                new_value=trimmed.iloc[index],
                rule="trim_whitespace",
                rationale="Removed leading/trailing whitespace to standardize categorical labels.",
            )

        empty_to_missing_mask = original.notna() & trimmed.isna()
        for index in df.index[empty_to_missing_mask]:
            add_change(
                changes=changes,
                row_number=index + 1,
                column=column,
                old_value=original.iloc[index],
                new_value="NA",
                rule="blank_to_missing",
                rationale="Blank string converted to missing value before categorical cleaning.",
            )

        df[column] = trimmed


def fill_missing_categories(df, changes):
    for column, fill_value in MISSING_IMPUTATION.items():
        missing_mask = df[column].isna()
        for index in df.index[missing_mask]:
            add_change(
                changes=changes,
                row_number=index + 1,
                column=column,
                old_value="NA",
                new_value=fill_value,
                rule="categorical_missing_to_unknown",
                rationale=(
                    "Missing category preserved explicitly as Unknown so downstream trees can split on missingness "
                    "instead of receiving an imputed guess."
                ),
            )
        df.loc[missing_mask, column] = fill_value


def clip_numeric_domains(df, changes):
    for column, (lower, upper) in NUMERIC_DOMAIN_BOUNDS.items():
        invalid_mask = (df[column] < lower) | (df[column] > upper)
        corrected = df[column].clip(lower=lower, upper=upper)
        for index in df.index[invalid_mask]:
            add_change(
                changes=changes,
                row_number=index + 1,
                column=column,
                old_value=df.at[index, column],
                new_value=corrected.at[index],
                rule="clip_to_domain",
                rationale=f"{column} is constrained to the domain [{lower}, {upper}].",
            )
        df[column] = corrected


def validate_categories(df):
    for column, allowed_values in EXPECTED_CATEGORIES.items():
        unexpected = sorted(set(df[column].dropna().unique()) - allowed_values)
        if unexpected:
            raise ValueError(
                f"Unexpected categories found in {column}: {unexpected}. "
                "Review the cleaning assumptions before using the output."
            )


def correlation_summary(df):
    numeric_df = df.select_dtypes(include="number")
    correlations = (
        numeric_df.corr(numeric_only=True)["Exam_Score"]
        .drop(labels=["Exam_Score"])
        .sort_values(ascending=False)
    )
    return correlations


def iqr_outlier_summary(df):
    rows = []
    numeric_columns = df.select_dtypes(include="number").columns
    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[column] < lower) | (df[column] > upper)).sum())
        rows.append(
            {
                "column": column,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": outlier_count,
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df):
    summaries = []
    categorical_columns = df.select_dtypes(exclude="number").columns
    for column in categorical_columns:
        counts = df[column].value_counts(dropna=False)
        summaries.append(f"[{column}]")
        summaries.append(counts.to_string())
        summaries.append("")
    return "\n".join(summaries).strip()


def write_report(original_df, clean_df, changes_df):
    missing_before = (
        original_df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda frame: (frame["missing_count"] / len(original_df) * 100).round(2))
    )
    missing_after = (
        clean_df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda frame: (frame["missing_count"] / len(clean_df) * 100).round(2))
    )
    numeric_summary = clean_df.describe().T[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    ].round(2)
    correlations = correlation_summary(clean_df).round(3)
    outlier_summary = iqr_outlier_summary(clean_df)
    categorical_distributions = categorical_summary(clean_df)
    change_summary = (
        changes_df.groupby(["rule", "column"]).size().rename("rows_changed").reset_index()
        if not changes_df.empty
        else pd.DataFrame(columns=["rule", "column", "rows_changed"])
    )

    report_lines = [
        "# Student Performance Factors: EDA and Cleaning Report",
        "",
        "## Dataset Overview",
        f"- Source file: `{INPUT_PATH.name}`",
        f"- Original shape: `{original_df.shape[0]} rows x {original_df.shape[1]} columns`",
        f"- Clean shape: `{clean_df.shape[0]} rows x {clean_df.shape[1]} columns`",
        f"- Exact duplicate rows before cleaning: `{int(original_df.duplicated().sum())}`",
        f"- Exact duplicate rows after cleaning: `{int(clean_df.duplicated().sum())}`",
        "",
        "## Cleaning Decisions",
        "- Standardized text labels by trimming whitespace.",
        "- Converted blank categorical values to missing before imputation.",
        "- Filled missing values in `Teacher_Quality`, `Parental_Education_Level`, and `Distance_from_Home` with `Unknown`.",
        "- Capped `Exam_Score` to the valid domain `[0, 100]`; only one row required correction (`101 -> 100`).",
        "- Did not delete statistical outliers identified by IQR because those rows remain domain-plausible and may carry real signal for tree-based models.",
        "",
        "## Missing Values Before Cleaning",
        "```text",
        missing_before.to_string(),
        "```",
        "",
        "## Missing Values After Cleaning",
        "```text",
        missing_after.to_string(),
        "```",
        "",
        "## Change Log Summary",
        "```text",
        change_summary.to_string(index=False) if not change_summary.empty else "No cell-level changes recorded.",
        "```",
        "",
        "## Numeric Summary After Cleaning",
        "```text",
        numeric_summary.to_string(),
        "```",
        "",
        "## Correlation With Exam_Score",
        "```text",
        correlations.to_string(),
        "```",
        "",
        "## Categorical Distributions After Cleaning",
        "```text",
        categorical_distributions,
        "```",
        "",
        "## IQR Outlier Scan",
        "```text",
        outlier_summary.to_string(index=False),
        "```",
        "",
        "## Notes for Decision Trees",
        "- Using `Unknown` keeps missingness explicit so a decision tree can branch on it.",
        "- No rows were removed, so sample size and class distribution remain intact.",
        "- Only domain-invalid values were corrected; statistical extremes were retained.",
    ]

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def main():
    original_df = pd.read_csv(INPUT_PATH)
    clean_df = original_df.copy()
    changes = []

    clean_df.columns = [column.strip() for column in clean_df.columns]
    trim_strings(clean_df, changes)
    fill_missing_categories(clean_df, changes)
    clip_numeric_domains(clean_df, changes)
    validate_categories(clean_df)

    changes_df = pd.DataFrame(changes)
    if not changes_df.empty:
        changes_df = changes_df.sort_values(["row_number", "column"]).reset_index(drop=True)

    clean_df.to_csv(OUTPUT_PATH, index=False)
    changes_df.to_csv(CHANGES_PATH, index=False)
    write_report(original_df=original_df, clean_df=clean_df, changes_df=changes_df)

    print(f"Clean CSV written to: {OUTPUT_PATH}")
    print(f"Cleaning change log written to: {CHANGES_PATH}")
    print(f"EDA report written to: {REPORT_PATH}")
    print(f"Rows changed: {len(changes_df)}")


if __name__ == "__main__":
    main()
