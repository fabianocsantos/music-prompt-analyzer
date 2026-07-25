from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DURACAO_SEGMENTO_SEGUNDOS,
)
from src.helpers import (
    formatar_tempo,
    normalizar_valores,
)


def limitar_valor(
    valor: float,
    minimo: float = 0.0,
    maximo: float = 1.0,
) -> float:
    """
    Mantém um valor dentro de um intervalo.
    """

    return float(
        max(
            minimo,
            min(
                maximo,
                valor,
            ),
        )
    )


def pontuar_faixa_triangular(
    valor: float,
    minimo: float,
    ideal: float,
    maximo: float,
) -> float:
    """
    Gera uma pontuação entre 0 e 1.

    A pontuação cresce até o valor ideal
    e diminui depois dele.
    """

    if valor <= minimo:
        return 0.0

    if valor >= maximo:
        return 0.0

    if valor == ideal:
        return 1.0

    if valor < ideal:
        intervalo = (
            ideal - minimo
        )

        if intervalo <= 0:
            return 0.0

        return limitar_valor(
            (valor - minimo)
            / intervalo
        )

    intervalo = (
        maximo - ideal
    )

    if intervalo <= 0:
        return 0.0

    return limitar_valor(
        (maximo - valor)
        / intervalo
    )


def calcular_energia(
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

    if rms.size == 0:
        return 0.0

    return float(
        np.mean(rms)
    )


def calcular_proporcao_harmonica(
    audio_harmonico: np.ndarray,
    audio_percussivo: np.ndarray,
) -> float:
    """
    Mede a proporção de energia harmônica de um trecho.
    """

    energia_harmonica = calcular_energia(
        audio_harmonico
    )

    energia_percussiva = calcular_energia(
        audio_percussivo
    )

    energia_total = (
        energia_harmonica
        + energia_percussiva
    )

    if energia_total <= 0:
        return 0.5

    return limitar_valor(
        energia_harmonica
        / energia_total
    )


def calcular_caracteristicas_trecho(
    trecho_audio: np.ndarray,
    trecho_harmonico: np.ndarray,
    trecho_percussivo: np.ndarray,
    taxa_amostragem: int,
) -> dict:
    """
    Extrai características usadas na estimativa vocal.
    """

    if trecho_audio.size == 0:
        return {
            "energia": 0.0,
            "centroide_hz": 0.0,
            "largura_hz": 0.0,
            "taxa_cruzamento_zero": 0.0,
            "planicidade_espectral": 1.0,
            "proporcao_harmonica": 0.5,
        }

    energia = calcular_energia(
        trecho_audio
    )

    centroide = (
        librosa.feature.spectral_centroid(
            y=trecho_harmonico,
            sr=taxa_amostragem,
        )
    )

    largura = (
        librosa.feature.spectral_bandwidth(
            y=trecho_harmonico,
            sr=taxa_amostragem,
        )
    )

    cruzamento_zero = (
        librosa.feature.zero_crossing_rate(
            trecho_audio
        )
    )

    planicidade = (
        librosa.feature.spectral_flatness(
            y=trecho_audio
        )
    )

    proporcao_harmonica = (
        calcular_proporcao_harmonica(
            audio_harmonico=trecho_harmonico,
            audio_percussivo=trecho_percussivo,
        )
    )

    return {
        "energia": float(
            np.mean(energia)
        ),
        "centroide_hz": float(
            np.mean(centroide)
        ),
        "largura_hz": float(
            np.mean(largura)
        ),
        "taxa_cruzamento_zero": float(
            np.mean(cruzamento_zero)
        ),
        "planicidade_espectral": float(
            np.mean(planicidade)
        ),
        "proporcao_harmonica": (
            proporcao_harmonica
        ),
    }


def classificar_probabilidade_vocal(
    pontuacao: float,
) -> str:
    """
    Converte a pontuação em uma classificação simples.
    """

    if pontuacao < 0.35:
        return "baixa"

    if pontuacao < 0.60:
        return "média"

    return "alta"


def calcular_pontuacao_vocal(
    caracteristicas: dict,
    energia_relativa: float,
    planicidade_relativa: float,
) -> dict:
    """
    Combina diferentes sinais acústicos em uma estimativa vocal.
    """

    pontuacao_centroide = (
        pontuar_faixa_triangular(
            valor=caracteristicas[
                "centroide_hz"
            ],
            minimo=250.0,
            ideal=1600.0,
            maximo=4500.0,
        )
    )

    pontuacao_largura = (
        pontuar_faixa_triangular(
            valor=caracteristicas[
                "largura_hz"
            ],
            minimo=300.0,
            ideal=1800.0,
            maximo=5000.0,
        )
    )

    pontuacao_cruzamento = (
        pontuar_faixa_triangular(
            valor=caracteristicas[
                "taxa_cruzamento_zero"
            ],
            minimo=0.01,
            ideal=0.08,
            maximo=0.25,
        )
    )

    pontuacao_harmonica = (
        limitar_valor(
            caracteristicas[
                "proporcao_harmonica"
            ]
        )
    )

    pontuacao_tonalidade = (
        1.0
        - limitar_valor(
            planicidade_relativa
        )
    )

    pontuacao = (
        energia_relativa * 0.25
        + pontuacao_harmonica * 0.20
        + pontuacao_tonalidade * 0.20
        + pontuacao_centroide * 0.15
        + pontuacao_cruzamento * 0.10
        + pontuacao_largura * 0.10
    )

    pontuacao = limitar_valor(
        pontuacao
    )

    return {
        "pontuacao": pontuacao,
        "pontuacao_energia": (
            energia_relativa
        ),
        "pontuacao_harmonica": (
            pontuacao_harmonica
        ),
        "pontuacao_tonalidade": (
            pontuacao_tonalidade
        ),
        "pontuacao_centroide": (
            pontuacao_centroide
        ),
        "pontuacao_cruzamento": (
            pontuacao_cruzamento
        ),
        "pontuacao_largura": (
            pontuacao_largura
        ),
    }


def analisar_segmentos_vocais(
    audio: np.ndarray,
    audio_harmonico: np.ndarray,
    audio_percussivo: np.ndarray,
    taxa_amostragem: int,
) -> list[dict]:
    """
    Analisa a probabilidade vocal em segmentos fixos.
    """

    duracao_total = float(
        librosa.get_duration(
            y=audio,
            sr=taxa_amostragem,
        )
    )

    segmentos_brutos = []
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

        trecho_audio = audio[
            inicio_amostra:fim_amostra
        ]

        trecho_harmonico = audio_harmonico[
            inicio_amostra:fim_amostra
        ]

        trecho_percussivo = audio_percussivo[
            inicio_amostra:fim_amostra
        ]

        caracteristicas = (
            calcular_caracteristicas_trecho(
                trecho_audio=trecho_audio,
                trecho_harmonico=trecho_harmonico,
                trecho_percussivo=trecho_percussivo,
                taxa_amostragem=taxa_amostragem,
            )
        )

        segmentos_brutos.append({
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
            **caracteristicas,
        })

        inicio_segmento = fim_segmento
        numero_segmento += 1

    energias = [
        segmento["energia"]
        for segmento in segmentos_brutos
    ]

    planicidades = [
        segmento["planicidade_espectral"]
        for segmento in segmentos_brutos
    ]

    energias_normalizadas = normalizar_valores(
        energias
    )

    planicidades_normalizadas = normalizar_valores(
        planicidades
    )

    segmentos_analisados = []

    for indice, segmento in enumerate(
        segmentos_brutos
    ):
        pontuacoes = calcular_pontuacao_vocal(
            caracteristicas=segmento,
            energia_relativa=(
                energias_normalizadas[indice]
            ),
            planicidade_relativa=(
                planicidades_normalizadas[indice]
            ),
        )

        pontuacao = pontuacoes[
            "pontuacao"
        ]

        classificacao = (
            classificar_probabilidade_vocal(
                pontuacao
            )
        )

        segmentos_analisados.append({
            "segmento": segmento[
                "segmento"
            ],
            "inicio_segundos": segmento[
                "inicio_segundos"
            ],
            "fim_segundos": segmento[
                "fim_segundos"
            ],
            "inicio_formatado": segmento[
                "inicio_formatado"
            ],
            "fim_formatado": segmento[
                "fim_formatado"
            ],
            "energia": round(
                segmento["energia"],
                6,
            ),
            "centroide_hz": round(
                segmento["centroide_hz"],
                2,
            ),
            "largura_hz": round(
                segmento["largura_hz"],
                2,
            ),
            "taxa_cruzamento_zero": round(
                segmento[
                    "taxa_cruzamento_zero"
                ],
                6,
            ),
            "planicidade_espectral": round(
                segmento[
                    "planicidade_espectral"
                ],
                6,
            ),
            "proporcao_harmonica": round(
                segmento[
                    "proporcao_harmonica"
                ],
                4,
            ),
            "probabilidade_vocal": round(
                pontuacao,
                4,
            ),
            "classificacao": classificacao,
            "pontuacao_energia": round(
                pontuacoes[
                    "pontuacao_energia"
                ],
                4,
            ),
            "pontuacao_harmonica": round(
                pontuacoes[
                    "pontuacao_harmonica"
                ],
                4,
            ),
            "pontuacao_tonalidade": round(
                pontuacoes[
                    "pontuacao_tonalidade"
                ],
                4,
            ),
            "pontuacao_centroide": round(
                pontuacoes[
                    "pontuacao_centroide"
                ],
                4,
            ),
            "pontuacao_cruzamento": round(
                pontuacoes[
                    "pontuacao_cruzamento"
                ],
                4,
            ),
            "pontuacao_largura": round(
                pontuacoes[
                    "pontuacao_largura"
                ],
                4,
            ),
        })

    return segmentos_analisados


def identificar_regioes_vocais(
    segmentos: list[dict],
) -> list[dict]:
    """
    Junta segmentos consecutivos com probabilidade média ou alta.
    """

    regioes = []

    for segmento in segmentos:
        classificacao = segmento[
            "classificacao"
        ]

        possui_indicio = (
            classificacao in {
                "média",
                "alta",
            }
        )

        if not possui_indicio:
            continue

        if (
            regioes
            and regioes[-1][
                "fim_segundos"
            ] == segmento[
                "inicio_segundos"
            ]
        ):
            regioes[-1]["fim_segundos"] = (
                segmento["fim_segundos"]
            )

            regioes[-1]["fim_formatado"] = (
                segmento["fim_formatado"]
            )

            regioes[-1]["segmentos"].append(
                segmento["segmento"]
            )

            regioes[-1]["pontuacoes"].append(
                segmento[
                    "probabilidade_vocal"
                ]
            )

        else:
            regioes.append({
                "inicio_segundos": segmento[
                    "inicio_segundos"
                ],
                "fim_segundos": segmento[
                    "fim_segundos"
                ],
                "inicio_formatado": segmento[
                    "inicio_formatado"
                ],
                "fim_formatado": segmento[
                    "fim_formatado"
                ],
                "segmentos": [
                    segmento["segmento"]
                ],
                "pontuacoes": [
                    segmento[
                        "probabilidade_vocal"
                    ]
                ],
            })

    for regiao in regioes:
        pontuacoes = regiao.pop(
            "pontuacoes"
        )

        regiao["pontuacao_media"] = round(
            float(
                np.mean(pontuacoes)
            ),
            4,
        )

        regiao["classificacao"] = (
            classificar_probabilidade_vocal(
                regiao["pontuacao_media"]
            )
        )

        regiao["duracao_segundos"] = round(
            regiao["fim_segundos"]
            - regiao["inicio_segundos"],
            2,
        )

    return regioes


def analisar_presenca_vocal(
    audio: np.ndarray,
    taxa_amostragem: int,
) -> dict:
    """
    Estima regiões com características compatíveis com voz.
    """

    print(
        "Estimando regiões com presença vocal provável..."
    )

    audio_harmonico, audio_percussivo = (
        librosa.effects.hpss(
            audio
        )
    )

    segmentos = analisar_segmentos_vocais(
        audio=audio,
        audio_harmonico=audio_harmonico,
        audio_percussivo=audio_percussivo,
        taxa_amostragem=taxa_amostragem,
    )

    regioes = identificar_regioes_vocais(
        segmentos
    )

    probabilidades = [
        segmento["probabilidade_vocal"]
        for segmento in segmentos
    ]

    probabilidade_media = (
        float(
            np.mean(probabilidades)
        )
        if probabilidades
        else 0.0
    )

    segmentos_alta = [
        segmento
        for segmento in segmentos
        if segmento["classificacao"] == "alta"
    ]

    segmentos_media = [
        segmento
        for segmento in segmentos
        if segmento["classificacao"] == "média"
    ]

    segmentos_baixa = [
        segmento
        for segmento in segmentos
        if segmento["classificacao"] == "baixa"
    ]

    melhor_segmento = (
        max(
            segmentos,
            key=lambda item: item[
                "probabilidade_vocal"
            ],
        )
        if segmentos
        else None
    )

    destaque = None

    if melhor_segmento:
        destaque = {
            "inicio": melhor_segmento[
                "inicio_formatado"
            ],
            "fim": melhor_segmento[
                "fim_formatado"
            ],
            "probabilidade": melhor_segmento[
                "probabilidade_vocal"
            ],
            "classificacao": melhor_segmento[
                "classificacao"
            ],
        }

    return {
        "metodo": (
            "heurística acústica baseada em energia, "
            "harmonicidade e características espectrais"
        ),
        "nao_separa_voz": True,
        "probabilidade_media": round(
            probabilidade_media,
            4,
        ),
        "classificacao_geral": (
            classificar_probabilidade_vocal(
                probabilidade_media
            )
        ),
        "segmentos_alta_probabilidade": len(
            segmentos_alta
        ),
        "segmentos_media_probabilidade": len(
            segmentos_media
        ),
        "segmentos_baixa_probabilidade": len(
            segmentos_baixa
        ),
        "destaque_principal": destaque,
        "regioes_provaveis": regioes,
        "linha_do_tempo": segmentos,
    }


def gerar_grafico_presenca_vocal(
    analise_vocal: dict,
    caminho_grafico: Path,
    nome_musica: str,
) -> None:
    """
    Gera o gráfico da probabilidade vocal ao longo do tempo.
    """

    segmentos = analise_vocal[
        "linha_do_tempo"
    ]

    tempos_centrais = [
        (
            segmento["inicio_segundos"]
            + segmento["fim_segundos"]
        ) / 2
        for segmento in segmentos
    ]

    probabilidades = [
        segmento["probabilidade_vocal"]
        for segmento in segmentos
    ]

    figura, eixo = plt.subplots(
        figsize=(14, 7)
    )

    eixo.plot(
        tempos_centrais,
        probabilidades,
        marker="o",
        linewidth=2,
        label="Probabilidade vocal",
    )

    eixo.fill_between(
        tempos_centrais,
        probabilidades,
        alpha=0.25,
    )

    eixo.axhline(
        y=0.35,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Limite médio",
    )

    eixo.axhline(
        y=0.60,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Limite alto",
    )

    eixo.set_title(
        f"Presença vocal provável: "
        f"{nome_musica}"
    )

    eixo.set_xlabel(
        "Tempo da música"
    )

    eixo.set_ylabel(
        "Probabilidade estimada"
    )

    eixo.set_ylim(
        0,
        1,
    )

    if segmentos:
        duracao_total = max(
            segmento["fim_segundos"]
            for segmento in segmentos
        )

        marcacoes = np.arange(
            0,
            duracao_total + 30,
            30,
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