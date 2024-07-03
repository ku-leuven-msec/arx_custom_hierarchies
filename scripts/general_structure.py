import pandas as pd


def get_parameters():
    """Print each param in expected order, and using the following structure:
    ['Name','Type','Default value','Description'].
    Finally print an overall description of the script as a String."""
    pass


def create_hierarchy(data:list[str]) -> pd.DataFrame:
    """Implementation of your custom hierarchy script and parameter parsing.
    Returns a dataframe representing the hierarchy."""
    pass


def print_hierarchy(hierarchy:pd.DataFrame):
    """Print the hierarchy using ',' as the column seperator."""
    for index, row in hierarchy.iterrows():
        print(','.join(row.tolist()))


if __name__ == '__main__':
    data = []
    data_size = int(input())
    for i in range(data_size):
        data.append(input())

    if (data[0] == "getParameters"):
        get_parameters()
    else:
        hierarchy = create_hierarchy(data)
        print_hierarchy(hierarchy)