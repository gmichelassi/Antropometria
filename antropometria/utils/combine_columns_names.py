import pandas as pd


def combine_columns_names(n_columns: int, columns_names: pd.Index, mode: str = 'default') -> list:
    names = []
    if mode == 'default':
        for i in range(0, n_columns):
            for j in range(i+1, n_columns):
                names.append(f"{i}/{j}")
    elif mode == 'complete':
        for i in range(0, n_columns):
            for j in range(i+1, n_columns):
                names.append(f"{columns_names[i]}/{columns_names[j]}")

    return names
