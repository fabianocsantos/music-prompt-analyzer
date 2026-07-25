from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DURACAO_SEGMENTO_SEGUNDOS,
    HOP_LENGTH,
    LIMIAR_CRESCIMENTO_FORTE,
    LIMIAR_MUDANCA_SECA,
    LIMIAR_QUEDA_FORTE,
)
from src.helpers import (
    formatar_tempo,
    normalizar_valores,
)


def classificar_dinamica_relativa(
    energia_normalizada: float,
) -> str:
    """
    Classifica a energia de um segmento em relação à própria música.
    """

    if energia_normalizada < 0.20:
        return "muito suave"

    if energia_normalizada < 0.40:
        return "suave"

    if energia_normalizada < 0.65:
        return "moderado"

    if energia_normalizada < 0.85:
        return "intenso"

    return "muito intenso"


def identificar_tendencia_dinamica(
    energias: list[float],
) -> str:
    """
    Compara o começo e o fim da faixa para estimar sua tendência.
    """

    if len(energias) < 3:
        return "estável"

    tamanho_grupo = max(
        1,
        len(energias) // 3,
    )

    energia_inicio = float(
        np.mean(
            energias[:tamanho_grupo]
        )
    )

    energia_final = float(
        np.mean(
            energias[-tamanho_grupo:]
        )
    )

    diferenca = (
        energia_final
        - energia_inicio
    )

    referencia = max(
        energia_inicio,
        energia_final,
        0.000001,
    )

    variacao_relativa = (
        diferenca / referencia
    )

    if variacao_relativa > 0.20:
        return "cresce ao longo da faixa"

    if variacao_relativa < -0.20:
        return "perde intensidade ao longo da faixa"

    return "mantém intensidade relativamente estável"


def calcular_media_segmento(
    valores: np.ndarray,
    mascara: np.ndarray,
) -> float:
    """
    Calcula a média dos valores pertencentes a um segmento.
    """

    trecho = valores[mascara]

    if trecho.size == 0:
        return 0.0

    return float(
        np.mean(trecho)
    )


def analisar_dinamica(
    audio: np.ndarray,
    taxa_amostragem: int,
) -> dict:
    """
    Divide a música em segmentos e analisa energia e brilho.
    """

    rms_frames = librosa.feature.rms(
        y=audio,
        hop_length=HOP_LENGTH,
    )[0]

    centroide_frames = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=taxa_amostragem,
            hop_length=HOP_LENGTH,
        )[0]
    )

    tempos_frames = librosa.frames_to_time(
        np.arange(
            len(rms_frames)
        ),
        sr=taxa_amostragem,
        hop_length=HOP_LENGTH,
    )

    duracao_total = float(
        librosa.get_duration(
            y=audio,
            sr=taxa_amostragem,
        )
    )

    segmentos_brutos = []
    inicio_segmento = 0.0

    while inicio_segmento < duracao_total:
        fim_segmento = min(
            inicio_segmento
            + DURACAO_SEGMENTO_SEGUNDOS,
            duracao_total,
        )

        mascara = (
            (tempos_frames >= inicio_segmento)
            & (tempos_frames < fim_segmento)
        )

        valores_rms = rms_frames[
            mascara
        ]

        if valores_rms.size == 0:
            energia_media = 0.0
            energia_maxima = 0.0

        else:
            energia_media = float(
                np.mean(
                    valores_rms
                )
            )

            energia_maxima = float(
                np.max(
                    valores_rms
                )
            )

        brilho_medio = calcular_media_segmento(
            centroide_frames,
            mascara,
        )

        segmentos_brutos.append({
            "inicio_segundos": round(
                inicio_segmento,
                2,
            ),
            "fim_segundos": round(
                fim_segmento,
                2,
            ),
            "energia_media_rms": round(
                energia_media,
                6,
            ),
            "energia_maxima_rms": round(
                energia_maxima,
                6,
            ),
            "brilho_medio_hz": round(
                brilho_medio,
                2,
            ),
        })

        inicio_segmento = fim_segmento

    energias_medias = [
        segmento["energia_media_rms"]
        for segmento in segmentos_brutos
    ]

    brilhos_medios = [
        segmento["brilho_medio_hz"]
        for segmento in segmentos_brutos
    ]

    energias_normalizadas = normalizar_valores(
        energias_medias
    )

    brilhos_normalizados = normalizar_valores(
        brilhos_medios
    )

    linha_do_tempo = []

    for indice, segmento in enumerate(
        segmentos_brutos
    ):
        energia_normalizada = (
            energias_normalizadas[indice]
        )

        brilho_normalizado = (
            brilhos_normalizados[indice]
        )

        linha_do_tempo.append({
            "segmento": indice + 1,
            "inicio_segundos": (
                segmento["inicio_segundos"]
            ),
            "fim_segundos": (
                segmento["fim_segundos"]
            ),
            "inicio_formatado": formatar_tempo(
                segmento["inicio_segundos"]
            ),
            "fim_formatado": formatar_tempo(
                segmento["fim_segundos"]
            ),
            "energia_media_rms": (
                segmento["energia_media_rms"]
            ),
            "energia_maxima_rms": (
                segmento["energia_maxima_rms"]
            ),
            "energia_relativa": round(
                energia_normalizada,
                4,
            ),
            "brilho_medio_hz": (
                segmento["brilho_medio_hz"]
            ),
            "brilho_relativo": round(
                brilho_normalizado,
                4,
            ),
            "classificacao": (
                classificar_dinamica_relativa(
                    energia_normalizada
                )
            ),
        })

    segmentos_ordenados = sorted(
        linha_do_tempo,
        key=lambda item: item[
            "energia_relativa"
        ],
        reverse=True,
    )

    picos = []

    for segmento in segmentos_ordenados[:3]:
        picos.append({
            "segmento": segmento[
                "segmento"
            ],
            "inicio": segmento[
                "inicio_formatado"
            ],
            "fim": segmento[
                "fim_formatado"
            ],
            "inicio_segundos": segmento[
                "inicio_segundos"
            ],
            "fim_segundos": segmento[
                "fim_segundos"
            ],
            "classificacao": segmento[
                "classificacao"
            ],
            "energia_relativa": segmento[
                "energia_relativa"
            ],
        })

    segmentos_suaves = [
        segmento
        for segmento in linha_do_tempo
        if segmento["energia_relativa"] < 0.40
    ]

    segmentos_intensos = [
        segmento
        for segmento in linha_do_tempo
        if segmento["energia_relativa"] >= 0.65
    ]

    energia_media_global = float(
        np.mean(
            energias_medias
        )
    )

    desvio_energia = float(
        np.std(
            energias_medias
        )
    )

    coeficiente_variacao = (
        desvio_energia
        / energia_media_global
        if energia_media_global > 0
        else 0.0
    )

    if coeficiente_variacao < 0.18:
        variacao_dinamica = "baixa"

    elif coeficiente_variacao < 0.35:
        variacao_dinamica = "moderada"

    else:
        variacao_dinamica = "alta"

    tendencia = identificar_tendencia_dinamica(
        energias_medias
    )

    return {
        "duracao_segmento_segundos": (
            DURACAO_SEGMENTO_SEGUNDOS
        ),
        "quantidade_segmentos": len(
            linha_do_tempo
        ),
        "variacao_dinamica": (
            variacao_dinamica
        ),
        "coeficiente_variacao": round(
            coeficiente_variacao,
            4,
        ),
        "tendencia_geral": tendencia,
        "segmentos_suaves": len(
            segmentos_suaves
        ),
        "segmentos_intensos": len(
            segmentos_intensos
        ),
        "picos_principais": picos,
        "linha_do_tempo": linha_do_tempo,
    }


def classificar_evento_transicao(
    variacao_energia: float,
    variacao_brilho: float,
    pontuacao_mudanca: float,
    energia_atual: float,
    energia_anterior: float,
) -> str:
    """
    Classifica uma mudança entre dois segmentos consecutivos.
    """

    if variacao_energia >= LIMIAR_CRESCIMENTO_FORTE:
        if (
            energia_anterior < 0.40
            and energia_atual >= 0.65
        ):
            return "entrada forte de nova seção"

        return "crescimento forte de intensidade"

    if variacao_energia <= LIMIAR_QUEDA_FORTE:
        if (
            energia_anterior >= 0.65
            and energia_atual < 0.40
        ):
            return "queda forte para trecho mais suave"

        return "queda forte de intensidade"

    if pontuacao_mudanca >= LIMIAR_MUDANCA_SECA:
        if abs(variacao_brilho) >= 0.35:
            return "possível mudança de seção e timbre"

        return "possível mudança de seção"

    if (
        energia_anterior < 0.40
        and energia_atual >= 0.55
    ):
        return "retomada de energia"

    return "mudança moderada"


def detectar_transicoes(
    dinamica: dict,
) -> list[dict]:
    """
    Detecta mudanças relevantes entre segmentos consecutivos.
    """

    linha_do_tempo = dinamica[
        "linha_do_tempo"
    ]

    transicoes = []

    for indice in range(
        1,
        len(linha_do_tempo),
    ):
        anterior = linha_do_tempo[
            indice - 1
        ]

        atual = linha_do_tempo[
            indice
        ]

        variacao_energia = (
            atual["energia_relativa"]
            - anterior["energia_relativa"]
        )

        variacao_brilho = (
            atual["brilho_relativo"]
            - anterior["brilho_relativo"]
        )

        pontuacao_mudanca = (
            abs(variacao_energia) * 0.70
            + abs(variacao_brilho) * 0.30
        )

        mudou_classificacao = (
            atual["classificacao"]
            != anterior["classificacao"]
        )

        evento_relevante = (
            abs(variacao_energia) >= 0.22
            or abs(variacao_brilho) >= 0.32
            or pontuacao_mudanca >= 0.30
        )

        if not evento_relevante:
            continue

        tipo_evento = classificar_evento_transicao(
            variacao_energia=variacao_energia,
            variacao_brilho=variacao_brilho,
            pontuacao_mudanca=pontuacao_mudanca,
            energia_atual=atual[
                "energia_relativa"
            ],
            energia_anterior=anterior[
                "energia_relativa"
            ],
        )

        transicoes.append({
            "tempo_segundos": atual[
                "inicio_segundos"
            ],
            "tempo_formatado": atual[
                "inicio_formatado"
            ],
            "segmento_anterior": anterior[
                "segmento"
            ],
            "segmento_atual": atual[
                "segmento"
            ],
            "evento": tipo_evento,
            "variacao_energia": round(
                variacao_energia,
                4,
            ),
            "variacao_brilho": round(
                variacao_brilho,
                4,
            ),
            "pontuacao_mudanca": round(
                pontuacao_mudanca,
                4,
            ),
            "mudou_classificacao": (
                mudou_classificacao
            ),
            "energia_anterior": anterior[
                "energia_relativa"
            ],
            "energia_atual": atual[
                "energia_relativa"
            ],
            "brilho_anterior": anterior[
                "brilho_relativo"
            ],
            "brilho_atual": atual[
                "brilho_relativo"
            ],
        })

    return transicoes


def gerar_grafico_dinamica(
    dinamica: dict,
    transicoes: list[dict],
    caminho_grafico: Path,
    nome_musica: str,
) -> None:
    """
    Gera o gráfico de energia e marca as transições detectadas.
    """

    linha_do_tempo = dinamica[
        "linha_do_tempo"
    ]

    tempos_centrais = [
        (
            segmento["inicio_segundos"]
            + segmento["fim_segundos"]
        ) / 2
        for segmento in linha_do_tempo
    ]

    energias = [
        segmento["energia_relativa"]
        for segmento in linha_do_tempo
    ]

    figura, eixo = plt.subplots(
        figsize=(14, 7)
    )

    eixo.plot(
        tempos_centrais,
        energias,
        marker="o",
        linewidth=2,
        label="Energia relativa",
    )

    eixo.fill_between(
        tempos_centrais,
        energias,
        alpha=0.25,
    )

    for pico in dinamica[
        "picos_principais"
    ]:
        tempo_central = (
            pico["inicio_segundos"]
            + pico["fim_segundos"]
        ) / 2

        eixo.scatter(
            tempo_central,
            pico["energia_relativa"],
            s=90,
            zorder=3,
        )

        eixo.annotate(
            f"Pico {pico['inicio']}",
            (
                tempo_central,
                pico["energia_relativa"],
            ),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    for transicao in transicoes:
        eixo.axvline(
            x=transicao[
                "tempo_segundos"
            ],
            linestyle=":",
            linewidth=1,
            alpha=0.65,
        )

    eixo.axhline(
        y=0.40,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    eixo.axhline(
        y=0.65,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )

    eixo.set_title(
        f"Dinâmica e transições: "
        f"{nome_musica}"
    )

    eixo.set_xlabel(
        "Tempo da música"
    )

    eixo.set_ylabel(
        "Energia relativa"
    )

    eixo.set_ylim(
        0,
        1.10,
    )

    duracao_total = max(
        segmento["fim_segundos"]
        for segmento in linha_do_tempo
    )

    intervalo_marcacao = 30

    marcacoes = np.arange(
        0,
        duracao_total
        + intervalo_marcacao,
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