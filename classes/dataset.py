import csv
import numpy as np


def load_dataset(path):
    """Le um .csv sem pandas.

    Retorna o cabecalho (lista de nomes de coluna) e os dados numericos
    como uma matriz numpy 2D. Linhas em branco sao ignoradas.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    data = np.array(rows, dtype=float)
    return header, data


def column(header, data, name):
    """Retorna a coluna `name` do dataset como vetor numpy."""
    return data[:, header.index(name)]
