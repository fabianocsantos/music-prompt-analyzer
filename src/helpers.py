import re

import numpy as np


def converter_para_numero(valor) -> float:
    """
    Converte números, valores NumPy ou arrays com um elemento para float.
    """

    array = np.asarray(valor)

    if array.size == 0:
        return 0.0

    return float(array.reshape(-1)[0])


def formatar_tempo(segundos: float) -> str:
    """
    Converte uma quantidade de segundos para o formato MM:SS.
    """

    segundos_inteiros = max(
        0,
        int(round(segundos)),
    )

    minutos = segundos_inteiros // 60
    segundos_restantes = segundos_inteiros % 60

    return (
        f"{minutos:02d}:"
        f"{segundos_restantes:02d}"
    )


def normalizar_valores(
    valores: list[float],
) -> list[float]:
    """
    Normaliza uma lista numérica para uma escala entre 0 e 1.

    Os percentis 10 e 90 reduzem o impacto de valores extremos.
    """

    if not valores:
        return []

    array = np.asarray(
        valores,
        dtype=float,
    )

    limite_inferior = float(
        np.percentile(array, 10)
    )

    limite_superior = float(
        np.percentile(array, 90)
    )

    diferenca = (
        limite_superior
        - limite_inferior
    )

    if diferenca <= 0:
        return [
            0.5
            for _ in valores
        ]

    normalizados = (
        array - limite_inferior
    ) / diferenca

    normalizados = np.clip(
        normalizados,
        0.0,
        1.0,
    )

    return [
        float(valor)
        for valor in normalizados
    ]


def normalizar_nome_arquivo(nome: str) -> str:
    """
    Transforma um nome em formato seguro para pastas e arquivos.
    """

    nome = nome.lower()

    substituicoes = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
    }

    for caractere, substituto in substituicoes.items():
        nome = nome.replace(
            caractere,
            substituto,
        )

    nome = re.sub(
        r"[^a-z0-9]+",
        "_",
        nome,
    )

    nome = nome.strip("_")

    if not nome:
        return "musica"

    return nome


def numero_para_rotulo(numero: int) -> str:
    """
    Converte números em letras estruturais.

    Exemplos:
    0 -> A
    1 -> B
    25 -> Z
    26 -> AA
    """

    rotulo = ""
    valor = numero

    while True:
        rotulo = (
            chr(ord("A") + valor % 26)
            + rotulo
        )

        valor = valor // 26 - 1

        if valor < 0:
            break

    return rotulo