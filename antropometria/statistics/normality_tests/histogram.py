import matplotlib.pyplot as plt
import pandas as pd


def generate_histogram(data: pd.Series, name: str = "histogram"):
    plt.hist(data)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f'{name}.png')
