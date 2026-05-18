from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, title: str, output_path: str | Path):
    """Guarda matriz de confusión como imagen."""
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_xticks([0, 1], labels=["Negativo", "Positivo"])
    ax.set_yticks([0, 1], labels=["Negativo", "Positivo"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_results_bar(results_df, output_path: str | Path):
    """Guarda gráfica de F1-score por configuración."""
    output_path = Path(output_path)
    pivot = results_df.pivot_table(
        index="configuracion",
        columns="modelo",
        values="f1_score",
        aggfunc="mean",
    )

    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_title("Comparación de F1-score por configuración")
    ax.set_xlabel("Representación + procesamiento")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Modelo")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
