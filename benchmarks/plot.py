"""Plot benchmarks/results/spark_vs_pandas.csv -> docs/img/spark_vs_pandas.png."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SERIES = [  # fixed categorical order: entity -> colour never changes
    ('pandas', 'Pandas (1 process)', '#2a78d6', 'o', -7),
    ('spark_rows', 'Spark, row-based (JVM reads grids)', '#eb6834', 's', 0),
    ('spark_files', 'Spark, file-based (Python reads grids)', '#1baf7a', '^', 7),
]


def main():
    df = pd.read_csv(REPO / 'benchmarks' / 'results' / 'spark_vs_pandas.csv')
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for engine, label, color, marker, dy in SERIES:
        d = df[df.engine == engine].sort_values('buildings')
        ax.plot(d.buildings, d.median_job_s, color=color, lw=2, marker=marker, ms=7,
                markeredgecolor='#fcfcfb', markeredgewidth=1.5, label=label)
        last = d.iloc[-1]
        ax.annotate(f'{last.median_job_s:.0f} s', (last.buildings, last.median_job_s),
                    xytext=(8, dy), textcoords='offset points', va='center', fontsize=9, color='#0b0b0b')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('buildings (≥200: replicas of 20 simulated floor plans)', color='#52514e')
    ax.set_ylabel('median job time, s (log)', color='#52514e')
    ax.set_title('raw grids → blocks → summary, M2 laptop, Spark local[4]', loc='left', fontsize=11, color='#0b0b0b')
    ax.set_xticks(df.buildings.unique())
    ax.set_xticklabels([f'{b:,}' for b in df.buildings.unique()])
    ax.set_xlim(right=df.buildings.max() * 1.6)
    ax.grid(True, which='major', color='#e6e5e0', lw=0.8)
    ax.minorticks_off()
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c2')
    ax.tick_params(colors='#52514e')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    out = REPO / 'docs' / 'img' / 'spark_vs_pandas.png'
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
