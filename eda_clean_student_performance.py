from pathlib import Path

import pandas as pd


INPUT_PATH = Path("StudentPerformanceFactors.csv")
OUTPUT_PATH = Path("StudentPerformanceFactors_clean.csv")
CHANGES_PATH = Path("StudentPerformanceFactors_cleaning_changes.csv")
REPORT_PATH = Path("StudentPerformanceFactors_eda_cleaning_report.md")

MISSING_IMPUTATION = {
    "Teacher_Quality": "Unknown",
    "Parental_Education_Level": "Unknown",
    "Distance_from_Home": "Unknown",
}

NUMERIC_DOMAIN_BOUNDS = {
    "Exam_Score": (0, 100),
}

REPORT_NUMERIC_COLUMNS = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")


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


def record_changes(df, changes, column, mask, *, rule, rationale, new_values, old_values=None):
    if not mask.any():
        return

    masked_old_values = df.loc[mask, column] if old_values is None else old_values
    if isinstance(new_values, pd.Series):
        if isinstance(masked_old_values, pd.Series):
            for index, old_value, new_value in zip(
                masked_old_values.index, masked_old_values, new_values.loc[mask]
            ):
                add_change(
                    changes=changes,
                    row_number=index + 1,
                    column=column,
                    old_value=old_value,
                    new_value=new_value,
                    rule=rule,
                    rationale=rationale,
                )
            return

        for index, new_value in new_values.loc[mask].items():
            add_change(
                changes=changes,
                row_number=index + 1,
                column=column,
                old_value=masked_old_values,
                new_value=new_value,
                rule=rule,
                rationale=rationale,
            )
        return

    if isinstance(masked_old_values, pd.Series):
        for index, old_value in masked_old_values.items():
            add_change(
                changes=changes,
                row_number=index + 1,
                column=column,
                old_value=old_value,
                new_value=new_values,
                rule=rule,
                rationale=rationale,
            )
        return

    for index in df.index[mask]:
        add_change(
            changes=changes,
            row_number=index + 1,
            column=column,
            old_value=masked_old_values,
            new_value=new_values,
            rule=rule,
            rationale=rationale,
        )


def trim_strings(df, changes):
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        original = df[column]
        cleaned = original.astype("string").str.strip().replace({"": pd.NA})

        trimmed_mask = original.notna() & cleaned.notna() & (
            original.astype(str) != cleaned.astype(str)
        )
        record_changes(
            df,
            changes,
            column,
            trimmed_mask,
            rule="trim_whitespace",
            rationale="Removed leading/trailing whitespace to standardize categorical labels.",
            new_values=cleaned,
        )

        blank_mask = original.notna() & cleaned.isna()
        record_changes(
            df,
            changes,
            column,
            blank_mask,
            rule="blank_to_missing",
            rationale="Blank string converted to missing value before categorical cleaning.",
            new_values="NA",
        )

        df[column] = cleaned


def fill_missing_categories(df, changes):
    for column, fill_value in MISSING_IMPUTATION.items():
        missing_mask = df[column].isna()
        record_changes(
            df,
            changes,
            column,
            missing_mask,
            rule="categorical_missing_to_unknown",
            rationale=(
                "Missing category preserved explicitly as Unknown so downstream trees can split on missingness "
                "instead of receiving an imputed guess."
            ),
            new_values=fill_value,
            old_values="NA",
        )
        df.loc[missing_mask, column] = fill_value


def clip_numeric_domains(df, changes):
    for column, (lower, upper) in NUMERIC_DOMAIN_BOUNDS.items():
        corrected = df[column].clip(lower=lower, upper=upper)
        invalid_mask = (df[column] < lower) | (df[column] > upper)
        record_changes(
            df,
            changes,
            column,
            invalid_mask,
            rule="clip_to_domain",
            rationale=f"{column} is constrained to the domain [{lower}, {upper}].",
            new_values=corrected,
        )
        df[column] = corrected


def missing_summary(df):
    summary = df.isna().sum().rename("missing_count").to_frame()
    row_count = max(len(df), 1)
    summary["missing_pct"] = (summary["missing_count"] / row_count * 100).round(2)
    return summary


def numeric_summary(df):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame(columns=REPORT_NUMERIC_COLUMNS)
    return numeric_df.describe().T.reindex(columns=REPORT_NUMERIC_COLUMNS).round(2)


def correlation_summary(df):
    correlations = df.select_dtypes(include="number").corr(numeric_only=True).get("Exam_Score")
    if correlations is None:
        return pd.Series(dtype=float)
    return correlations.drop(labels=["Exam_Score"], errors="ignore").sort_values(ascending=False)


def iqr_outlier_summary(df):
    rows = []
    for column, series in df.select_dtypes(include="number").items():
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "column": column,
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "outlier_count": int(((series < lower) | (series > upper)).sum()),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df):
    sections = [
        f"[{column}]\n{df[column].value_counts(dropna=False).to_string()}"
        for column in df.select_dtypes(exclude="number").columns
    ]
    return "\n\n".join(sections).strip()


def change_summary_table(changes_df):
    if changes_df.empty:
        return pd.DataFrame(columns=["rule", "column", "rows_changed"])
    return changes_df.groupby(["rule", "column"]).size().rename("rows_changed").reset_index()


def changes_dataframe(changes):
    if not changes:
        return pd.DataFrame()
    return pd.DataFrame(changes).sort_values(["row_number", "column"]).reset_index(drop=True)


def append_text_section(lines, title, content):
    lines.extend([title, "```text", content, "```", ""])


def join_with_and(items):
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def cleaning_decisions(changes_df):
    fill_columns = join_with_and(f"`{column}`" for column in MISSING_IMPUTATION)
    decisions = [
        "- Standardized text labels by trimming whitespace.",
        "- Converted blank categorical values to missing before imputation.",
        f"- Filled missing values in {fill_columns} with `Unknown`.",
    ]

    for column, (lower, upper) in NUMERIC_DOMAIN_BOUNDS.items():
        if changes_df.empty:
            decisions.append(
                f"- Capped `{column}` to the valid domain `[{lower}, {upper}]`; no rows required correction."
            )
            continue

        clipped = changes_df[
            (changes_df["rule"] == "clip_to_domain") & (changes_df["column"] == column)
        ]

        if clipped.empty:
            decisions.append(
                f"- Capped `{column}` to the valid domain `[{lower}, {upper}]`; no rows required correction."
            )
            continue

        unique_pairs = clipped[["old_value", "new_value"]].drop_duplicates()
        if len(clipped) == 1 and len(unique_pairs) == 1:
            old_value, new_value = unique_pairs.iloc[0]
            decisions.append(
                f"- Capped `{column}` to the valid domain `[{lower}, {upper}]`; only one row required correction "
                f"(`{old_value} -> {new_value}`)."
            )
        else:
            decisions.append(
                f"- Capped `{column}` to the valid domain `[{lower}, {upper}]`; {len(clipped)} rows required "
                "correction."
            )

    decisions.append(
        "- Did not delete statistical outliers identified by IQR because those rows remain domain-plausible and "
        "may carry real signal for tree-based models."
    )
    return decisions


def write_report(original_df, clean_df, changes_df):
    missing_before = missing_summary(original_df)
    missing_after = missing_summary(clean_df)
    numeric_after = numeric_summary(clean_df)
    correlations = correlation_summary(clean_df).round(3)
    outlier_summary = iqr_outlier_summary(clean_df)
    categorical_distributions = categorical_summary(clean_df)
    change_summary = change_summary_table(changes_df)

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
    ]
    report_lines.extend(cleaning_decisions(changes_df))
    report_lines.append("")

    append_text_section(report_lines, "## Missing Values Before Cleaning", missing_before.to_string())
    append_text_section(report_lines, "## Missing Values After Cleaning", missing_after.to_string())
    append_text_section(
        report_lines,
        "## Change Log Summary",
        change_summary.to_string(index=False) if not change_summary.empty else "No cell-level changes recorded.",
    )
    append_text_section(
        report_lines,
        "## Numeric Summary After Cleaning",
        numeric_after.to_string(),
    )
    append_text_section(report_lines, "## Correlation With Exam_Score", correlations.to_string())
    append_text_section(
        report_lines,
        "## Categorical Distributions After Cleaning",
        categorical_distributions,
    )
    append_text_section(report_lines, "## IQR Outlier Scan", outlier_summary.to_string(index=False))

    report_lines.extend(
        [
            "## Notes for Decision Trees",
            "- Using `Unknown` keeps missingness explicit so a decision tree can branch on it.",
            "- No rows were removed, so sample size and class distribution remain intact.",
            "- Only domain-invalid values were corrected; statistical extremes were retained.",
        ]
    )

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def clean_dataset(original_df):
    clean_df = original_df.copy()
    clean_df.columns = clean_df.columns.str.strip()

    changes = []
    for step in (trim_strings, fill_missing_categories, clip_numeric_domains):
        step(clean_df, changes)

    return clean_df, changes_dataframe(changes)


def main():
    original_df = pd.read_csv(INPUT_PATH)
    clean_df, changes_df = clean_dataset(original_df)

    clean_df.to_csv(OUTPUT_PATH, index=False)
    changes_df.to_csv(CHANGES_PATH, index=False)
    write_report(original_df=original_df, clean_df=clean_df, changes_df=changes_df)

    print(f"Clean CSV written to: {OUTPUT_PATH}")
    print(f"Cleaning change log written to: {CHANGES_PATH}")
    print(f"EDA report written to: {REPORT_PATH}")
    print(f"Rows changed: {len(changes_df)}")


if __name__ == "__main__":
    main()
