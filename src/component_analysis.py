from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DURACAO_SEGMENTO_SEGUNDOS,
)
from src.helpers import (
    formatar_tempo,
)


def calcular_energia_audio(
    audio: np.ndarray,
) -> float:
    """
    Calcula a energia média RMS de um sinal.
    """

    if audio.size == 0:
        return 0.0

    rms = librosa.feature.rms(
        y=audio
    )

    return float(
        np.mean(rms)
    )


def classificar_equilibrio_componentes(
    proporcao_harmonica: float,
    proporcao_percussiva: float,
) -> str:
    """
    Classifica o equilíbrio geral entre conteúdo harmônico e percussivo.
    """

    diferenca = (
        proporcao_harmonica
        - proporcao_percussiva
    )

    if diferenca >= 0.30:
        return "predominantemente harmônica"

    if diferenca >= 0.10:
        return "mais harmônica que percussiva"

    if diferenca <= -0.30:
        return "predominantemente percussiva"

    if diferenca <= -0.10:
        return "mais percussiva que harmônica"

    return "equilibrada entre harmonia e percussão"


def classificar_segmento_componentes(
    proporcao_harmonica: float,
    proporcao_percussiva: float,
) -> str:
    """
    Classifica um segmento individual.
    """

    diferenca = (
        proporcao_harmonica
        - proporcao_percussiva
    )

    if diferenca >= 0.25:
        return "harmônico"

    if diferenca <= -0.25:
        return "percussivo"

    return "equilibrado"


def calcular_proporcoes(
    energia_harmonica: float,
    energia_percussiva: float,
) -> tuple[float, float]:
    """
    Converte as energias em proporções entre 0 e 1.
    """

    energia_total = (
        energia_harmonica
        + energia_percussiva
    )

    if energia_total <= 0:
        return 0.5, 0.5

    proporcao_harmonica = (
        energia_harmonica
        / energia_total
    )

    proporcao_percussiva = (
        energia_percussiva
        / energia_total
    )

    return (
        float(proporcao_harmonica),
        float(proporcao_percussiva),
    )


def analisar_segmentos_componentes(
    audio_harmonico: np.ndarray,
    audio_percussivo: np.ndarray,
    taxa_amostragem: int,
) -> list[dict]:
    """
    Analisa o equilíbrio harmônico e percussivo em blocos de tempo.
    """

    duracao_total = float(
        librosa.get_duration(
            y=audio_harmonico,
            sr=taxa_amostragem,
        )
    )

    segmentos = []
    inicio_segmento = 0.0
    numero_segmento = 1

    while inicio_segmento < duracao_total:
        fim_segmento = min(
            inicio_segmento
            + DURACAO_SEGMENTO_SEGUNDOS,
            duracao_total,
        )

        inicio_amostra = int(
            inicio_segmento
            * taxa_amostragem
        )

        fim_amostra = int(
            fim_segmento
            * taxa_amostragem
        )

        trecho_harmonico = audio_harmonico[
            inicio_amostra:fim_amostra
        ]

        trecho_percussivo = audio_percussivo[
            inicio_amostra:fim_amostra
        ]

        energia_harmonica = calcular_energia_audio(
            trecho_harmonico
        )

        energia_percussiva = calcular_energia_audio(
            trecho_percussivo
        )

        (
            proporcao_harmonica,
            proporcao_percussiva,
        ) = calcular_proporcoes(
            energia_harmonica=energia_harmonica,
            energia_percussiva=energia_percussiva,
        )

        classificacao = classificar_segmento_componentes(
            proporcao_harmonica=proporcao_harmonica,
            proporcao_percussiva=proporcao_percussiva,
        )

        segmentos.append({
            "segmento": numero_segmento,
            "inicio_segundos": round(
                inicio_segmento,
                2,
            ),
            "fim_segundos": round(
                fim_segmento,
                2,
            ),
            "inicio_formatado": formatar_tempo(
                inicio_segmento
            ),
            "fim_formatado": formatar_tempo(
                fim_segmento
            ),
            "energia_harmonica": round(
                energia_harmonica,
                6,
            ),
            "energia_percussiva": round(
                energia_percussiva,
                6,
            ),
            "proporcao_harmonica": round(
                proporcao_harmonica,
                4,
            ),
            "proporcao_percussiva": round(
                proporcao_percussiva,
                4,
            ),
            "classificacao": classificacao,
        })

        inicio_segmento = fim_segmento
        numero_segmento += 1

    return segmentos


def identificar_trechos_destaque(
    segmentos: list[dict],
) -> dict:
    """
    Localiza os trechos mais harmônicos e mais percussivos.
    """

    if not segmentos:
        return {
            "mais_harmonico": None,
            "mais_percussivo": None,
        }

    mais_harmonico = max(
        segmentos,
        key=lambda item: item[
            "proporcao_harmonica"
        ],
    )

    mais_percussivo = max(
        segmentos,
        key=lambda item: item[
            "proporcao_percussiva"
        ],
    )

    return {
        "mais_harmonico": {
            "inicio": mais_harmonico[
                "inicio_formatado"
            ],
            "fim": mais_harmonico[
                "fim_formatado"
            ],
            "proporcao": mais_harmonico[
                "proporcao_harmonica"
            ],
        },
        "mais_percussivo": {
            "inicio": mais_percussivo[
                "inicio_formatado"
            ],
            "fim": mais_percussivo[
                "fim_formatado"
            ],
            "proporcao": mais_percussivo[
                "proporcao_percussiva"
            ],
        },
    }


def analisar_componentes(
    audio: np.ndarray,
    taxa_amostragem: int,
) -> dict:
    """
    Separa e analisa conteúdo harmônico e percussivo.
    """

    print(
        "Separando conteúdo harmônico "
        "e percussivo em detalhes..."
    )

    audio_harmonico, audio_percussivo = (
        librosa.effects.hpss(
            audio
        )
    )

    energia_harmonica = calcular_energia_audio(
        audio_harmonico
    )

    energia_percussiva = calcular_energia_audio(
        audio_percussivo
    )

    (
        proporcao_harmonica,
        proporcao_percussiva,
    ) = calcular_proporcoes(
        energia_harmonica=energia_harmonica,
        energia_percussiva=energia_percussiva,
    )

    classificacao = classificar_equilibrio_componentes(
        proporcao_harmonica=proporcao_harmonica,
        proporcao_percussiva=proporcao_percussiva,
    )

    segmentos = analisar_segmentos_componentes(
        audio_harmonico=audio_harmonico,
        audio_percussivo=audio_percussivo,
        taxa_amostragem=taxa_amostragem,
    )

    destaques = identificar_trechos_destaque(
        segmentos
    )

    quantidade_harmonicos = sum(
        1
        for segmento in segmentos
        if segmento["classificacao"] == "harmônico"
    )

    quantidade_percussivos = sum(
        1
        for segmento in segmentos
        if segmento["classificacao"] == "percussivo"
    )

    quantidade_equilibrados = sum(
        1
        for segmento in segmentos
        if segmento["classificacao"] == "equilibrado"
    )

    return {
        "energia_harmonica": round(
            energia_harmonica,
            6,
        ),
        "energia_percussiva": round(
            energia_percussiva,
            6,
        ),
        "proporcao_harmonica": round(
            proporcao_harmonica,
            4,
        ),
        "proporcao_percussiva": round(
            proporcao_percussiva,
            4,
        ),
        "percentual_harmonico": round(
            proporcao_harmonica * 100,
            2,
        ),
        "percentual_percussivo": round(
            proporcao_percussiva * 100,
            2,
        ),
        "classificacao": classificacao,
        "segmentos_harmonicos": quantidade_harmonicos,
        "segmentos_percussivos": quantidade_percussivos,
        "segmentos_equilibrados": quantidade_equilibrados,
        "destaques": destaques,
        "linha_do_tempo": segmentos,
    }


def gerar_grafico_componentes(
    componentes: dict,
    caminho_grafico: Path,
    nome_musica: str,
) -> None:
    """
    Gera um gráfico com as proporções harmônica e percussiva.
    """

    segmentos = componentes[
        "linha_do_tempo"
    ]

    tempos_centrais = [
        (
            segmento["inicio_segundos"]
            + segmento["fim_segundos"]
        ) / 2
        for segmento in segmentos
    ]

    proporcoes_harmonicas = [
        segmento["proporcao_harmonica"]
        for segmento in segmentos
    ]

    proporcoes_percussivas = [
        segmento["proporcao_percussiva"]
        for segmento in segmentos
    ]

    figura, eixo = plt.subplots(
        figsize=(14, 7)
    )

    eixo.plot(
        tempos_centrais,
        proporcoes_harmonicas,
        marker="o",
        linewidth=2,
        label="Conteúdo harmônico",
    )

    eixo.plot(
        tempos_centrais,
        proporcoes_percussivas,
        marker="o",
        linewidth=2,
        label="Conteúdo percussivo",
    )

    eixo.axhline(
        y=0.5,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    eixo.set_title(
        f"Equilíbrio harmônico e percussivo: "
        f"{nome_musica}"
    )

    eixo.set_xlabel(
        "Tempo da música"
    )

    eixo.set_ylabel(
        "Proporção relativa"
    )

    eixo.set_ylim(
        0,
        1,
    )

    duracao_total = max(
        segmento["fim_segundos"]
        for segmento in segmentos
    )

    intervalo_marcacao = 30

    marcacoes = np.arange(
        0,
        duracao_total + intervalo_marcacao,
        intervalo_marcacao,
    )

    eixo.set_xticks(
        marcacoes
    )

    eixo.set_xticklabels([
        formatar_tempo(
            tempo
        )
        for tempo in marcacoes
    ])

    eixo.grid(
        alpha=0.25
    )

    eixo.legend()

    figura.tight_layout()

    figura.savefig(
        caminho_grafico,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figura
    )