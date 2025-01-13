import pandas as pd


def count_total_duplicates(row: pd.Series) -> int:
    """Return total duplicate entries (each extra occurrence) in a row."""
    return len(row) - row.nunique()

def count_total_unique(row: pd.Series) -> int:
    """Return total unique values in a row."""
    return row.nunique()


def count_distinct_duplicate_values(row: pd.Series) -> int:
    """Return the number of distinct values that occur more than once in a row."""
    return (row.value_counts() > 1).sum()


def count_missing_values(row: pd.Series) -> int:
    """Return the number of missing (NaN) values in a row."""
    return row.isna().sum()


def duplicate_mask(row: pd.Series) -> list:
    """Return a boolean list indicating duplicate positions in a row."""
    return list(row.duplicated(keep=False))


def list_duplicates_with_count(row: pd.Series) -> dict:
    """Return a dict of duplicate values and their counts (only values with count > 1) in a row."""
    counts = row.value_counts()
    return {value: count for value, count in counts.items() if count > 1}


def has_duplicates(row: pd.Series) -> bool:
    """Return True if the row contains any duplicates, otherwise False."""
    return row.duplicated(keep=False).any()


def missing_ratio(row: pd.Series) -> float:
    """Return the ratio of missing values in a row as a float between 0 and 1."""
    return count_missing_values(row) / len(row) if len(row) > 0 else 0.0

def top_n_duplicates(row: pd.Series, n: int = 5) -> dict:
    """Return a dictionary of the top n duplicate values in a row sotred."""
    dups = list_duplicates_with_count(row)
    sorted_dups = sorted(dups.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_dups[:n])

def print_column_checks(df: pd.DataFrame, max_return: int=20) -> None:
    """
    For each column in the DataFrame, print out various checks.

        - Total duplicate count (total extra occurrences)
        - Count of distinct duplicate values
        - Count of total unique values
        - Count of missing (NaN) values
        - Dictionary of each duplicate with its count
    """
    for col in df.columns:
        print(f"Column: {col}")
        print("  dups:", count_total_duplicates(df[col]))
        print("  distin:", count_distinct_duplicate_values(df[col]))
        print("  uniq:", count_total_unique(df[col]))
        print("  miss:", count_missing_values(df[col]))
        print("  count:", top_n_duplicates(df[col], n=max_return))
        print("-" * 40)
