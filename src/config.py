import numpy as np


EXTENSOES_SUPORTADAS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
}


NOMES_NOTAS = [
    "Dó",
    "Dó sustenido",
    "Ré",
    "Ré sustenido",
    "Mi",
    "Fá",
    "Fá sustenido",
    "Sol",
    "Sol sustenido",
    "Lá",
    "Lá sustenido",
    "Si",
]


PERFIL_MAIOR = np.array([
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
])


PERFIL_MENOR = np.array([
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
])


DURACAO_SEGMENTO_SEGUNDOS = 10
HOP_LENGTH = 512


LIMIAR_CRESCIMENTO_FORTE = 0.30
LIMIAR_QUEDA_FORTE = -0.30
LIMIAR_MUDANCA_SECA = 0.42


LIMIAR_SIMILARIDADE_DISTANTE = 0.84
LIMIAR_SIMILARIDADE_ADJACENTE = 0.92