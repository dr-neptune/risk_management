import pandas as pd
import numpy as np
from rich import print
from rich.table import Table
from rich.text import Text
from rich.console import Console


def allocation_contribution(sector) -> float:
    """
    Calculate the allocation contribution to active return
    :param sector: str
    :return: float
    """
    return (sector['portfolio weight'] - sector['benchmark weight']) * sector['benchmark return']


def selection_contribution(sector) -> float:
    """
    Calculate the selection allocation to active return
    :param sector: str
    :return: float
    """
    return sector['benchmark weight'] * (sector['portfolio return'] - sector['benchmark return'])


def interaction_contribution(sector) -> float:
    """
    Calculate the interaction contribution to active return
    :param sector: str
    :return: float
    """
    return (sector['portfolio weight'] - sector['benchmark weight']) * (sector['portfolio return'] - sector['benchmark return'])


def calculate_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the allocation, selection, and interaction contributions for each sector.
    :param df: DataFrame
    :return: DataFrame
    """
    # Calculate contributions
    df['allocation contribution'] = df.apply(allocation_contribution, axis=1)
    df['selection contribution'] = df.apply(selection_contribution, axis=1)
    df['interaction contribution'] = df.apply(interaction_contribution, axis=1)

    # Calculate contribution amounts for a $1,000,000 fund
    df['allocation amount'] = df['allocation contribution'] * 1000000
    df['selection amount'] = df['selection contribution'] * 1000000
    df['interaction amount'] = df['interaction contribution'] * 1000000

    # Calculate overall contributions and return
    df['overall allocation'] = df['allocation contribution'].sum()
    df['overall selection'] = df['selection contribution'].sum()
    df['overall interaction'] = df['interaction contribution'].sum()
    df['overall return'] = (df['portfolio return'] * df['portfolio weight']).sum()

    df['overall allocation amount'] = df['allocation amount'].sum()
    df['overall selection amount'] = df['selection amount'].sum()
    df['overall interaction amount'] = df['interaction amount'].sum()
    df['overall return amount'] = df['overall return'] * 1000000

    df['overall benchmark return'] = (df['benchmark return'] * df['benchmark weight']).sum()
    df['overall benchmark return amount'] = df['overall benchmark return'] * 1000000

    df['overall difference'] = df['overall return'] - df['overall benchmark return']
    df['overall difference amount'] = df['overall difference'] * 1000000

    return df


def print_contributions(df: pd.DataFrame) -> None:
    """
    Print the contributions and returns for each sector and overall.
    :param df: DataFrame
    """
    console = Console()

    for i, row in df.iterrows():
        table = Table(show_header=True, header_style="bold magenta")
        title = Text.assemble((f"{row['sector']}", "bold blue"), ("\nFor a fund of $1,000,000", "normal"))
        table.title = title

        table.add_column("Contribution Type", justify="left")
        table.add_column("Value", justify="right")
        table.add_column("Amount ($)", justify="right")

        table.add_row("Allocation Contribution", f"{row['allocation contribution']*100:.2f}%", f"${row['allocation amount']:,.2f}")
        table.add_row("Selection Contribution", f"{row['selection contribution']*100:.2f}%", f"${row['selection amount']:,.2f}")
        table.add_row("Interaction Contribution", f"{row['interaction contribution']*100:.2f}%", f"${row['interaction amount']:,.2f}")

        console.print(table)
        console.print("\n")

    table = Table(show_header=True, header_style="bold magenta")
    title = Text.assemble(("Overall Contributions and Return", "bold green"), ("\nFor a fund of $1,000,000", "normal"))
    table.title = title

    table.add_column("Contribution Type", justify="left")
    table.add_column("Value", justify="right")
    table.add_column("Amount ($)", justify="right")

    table.add_row("Overall Allocation Contribution", f"{row['overall allocation']*100:.2f}%", f"${row['overall allocation amount']:,.2f}")
    table.add_row("Overall Selection Contribution", f"{row['overall selection']*100:.2f}%", f"${row['overall selection amount']:,.2f}")
    table.add_row("Overall Interaction Contribution", f"{row['overall interaction']*100:.2f}%", f"${row['overall interaction amount']:,.2f}")
    table.add_row("Overall Portfolio Return", f"{row['overall return']*100:.2f}%", f"${row['overall return amount']:,.2f}")
    table.add_row("Overall Benchmark Return", f"{row['overall benchmark return']*100:.2f}%", f"${row['overall benchmark return amount']:,.2f}")
    table.add_row("Difference Portfolio-Benchmark", f"{row['overall difference']*100:.2f}%", f"${row['overall difference amount']:,.2f}")

    console.print(table)


if __name__ == '__main__':
    # List of sectors
    sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
               'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
               'Utilities', 'Real Estate', 'Materials']

    # Number of sectors
    n_sectors = len(sectors)

    # Create a pandas DataFrame
    df = pd.DataFrame({
        'sector': sectors,
        'portfolio weight': np.random.dirichlet(np.ones(n_sectors),size=1)[0],  # Ensure weights sum to 1
        'benchmark weight': np.random.dirichlet(np.ones(n_sectors),size=1)[0],  # Ensure weights sum to 1
        'portfolio return': np.random.uniform(0, 1, n_sectors),
        'benchmark return': np.random.uniform(0, 1, n_sectors)
    })

    # Apply the functions
    df = calculate_contributions(df)
    print_contributions(df)
