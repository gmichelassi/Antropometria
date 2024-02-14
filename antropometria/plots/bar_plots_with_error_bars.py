import matplotlib.pyplot as plt


def generate_bar_plots_with_error_bars(x: list, y: list, errors: list, output_file: str) -> None:
    plt.figure(figsize=(10, 10), tight_layout=True)
    container = plt.bar(x, y, color='#8FBFE0', edgecolor='#083D77', linewidth=2)
    plt.bar_label(container, labels=y, label_type='center', color='#FFF', fontweight='bold', fontsize=16)

    plt.xticks(rotation=-15, fontsize=12)
    plt.yticks(fontsize=16)

    plt.errorbar(x, y, yerr=errors, fmt="o", color="#DA4167", elinewidth=5, capsize=10)

    plt.savefig(output_file)
