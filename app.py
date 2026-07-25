import csv
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.audio_files import (
    escolher_arquivo_audio,
    listar_arquivos_audio,
)
from src.basic_analysis import (
    analisar_caracteristicas_basicas,
)
from src.config import (
    DURACAO_SEGMENTO_SEGUNDOS,
    HOP_LENGTH,
    LIMIAR_CRESCIMENTO_FORTE,
    LIMIAR_MUDANCA_SECA,
    LIMIAR_QUEDA_FORTE,
    LIMIAR_SIMILARIDADE_ADJACENTE,
    LIMIAR_SIMILARIDADE_DISTANTE,
)
from src.helpers import (
    formatar_tempo,
    normalizar_nome_arquivo,
    normalizar_valores,
    numero_para_rotulo,
)


def classificar_dinamica_relativa(
    energia_normalizada: float,
) -> str:
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


def extrair_vetor_segmento(
    trecho_audio: np.ndarray,
    taxa_amostragem: int,
) -> np.ndarray:
    if trecho_audio.size == 0:
        return np.zeros(49)

    chroma = librosa.feature.chroma_cqt(
        y=trecho_audio,
        sr=taxa_amostragem,
    )

    mfcc = librosa.feature.mfcc(
        y=trecho_audio,
        sr=taxa_amostragem,
        n_mfcc=13,
    )

    contraste = (
        librosa.feature.spectral_contrast(
            y=trecho_audio,
            sr=taxa_amostragem,
        )
    )

    rms = librosa.feature.rms(
        y=trecho_audio
    )

    centroide = (
        librosa.feature.spectral_centroid(
            y=trecho_audio,
            sr=taxa_amostragem,
        )
    )

    largura = (
        librosa.feature.spectral_bandwidth(
            y=trecho_audio,
            sr=taxa_amostragem,
        )
    )

    zero_crossing = (
        librosa.feature.zero_crossing_rate(
            trecho_audio
        )
    )

    vetor = np.concatenate([
        np.mean(chroma, axis=1),
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(contraste, axis=1),
        np.array([
            float(np.mean(rms)),
            float(np.mean(centroide)),
            float(np.mean(largura)),
            float(np.mean(zero_crossing)),
        ]),
    ])

    return vetor.astype(float)


def padronizar_vetores(
    vetores: np.ndarray,
) -> np.ndarray:
    medias = np.mean(
        vetores,
        axis=0,
    )

    desvios = np.std(
        vetores,
        axis=0,
    )

    desvios[desvios == 0] = 1.0

    padronizados = (
        vetores - medias
    ) / desvios

    normas = np.linalg.norm(
        padronizados,
        axis=1,
        keepdims=True,
    )

    normas[normas == 0] = 1.0

    return padronizados / normas


def criar_matriz_similaridade(
    audio: np.ndarray,
    taxa_amostragem: int,
    linha_do_tempo: list[dict],
) -> np.ndarray:
    vetores = []

    for segmento in linha_do_tempo:
        inicio_amostra = int(
            segmento["inicio_segundos"]
            * taxa_amostragem
        )

        fim_amostra = int(
            segmento["fim_segundos"]
            * taxa_amostragem
        )

        trecho_audio = audio[
            inicio_amostra:fim_amostra
        ]

        vetor = extrair_vetor_segmento(
            trecho_audio,
            taxa_amostragem,
        )

        vetores.append(vetor)

    matriz_vetores = np.asarray(
        vetores,
        dtype=float,
    )

    vetores_padronizados = padronizar_vetores(
        matriz_vetores
    )

    similaridade = (
        vetores_padronizados
        @ vetores_padronizados.T
    )

    return np.clip(
        similaridade,
        -1.0,
        1.0,
    )


class Agrupador:
    def __init__(
        self,
        quantidade: int,
    ):
        self.pais = list(
            range(quantidade)
        )

    def encontrar(
        self,
        indice: int,
    ) -> int:
        if self.pais[indice] != indice:
            self.pais[indice] = self.encontrar(
                self.pais[indice]
            )

        return self.pais[indice]

    def unir(
        self,
        primeiro: int,
        segundo: int,
    ) -> None:
        raiz_primeiro = self.encontrar(
            primeiro
        )

        raiz_segundo = self.encontrar(
            segundo
        )

        if raiz_primeiro != raiz_segundo:
            self.pais[raiz_segundo] = (
                raiz_primeiro
            )


def agrupar_segmentos_similares(
    matriz_similaridade: np.ndarray,
) -> list[str]:
    quantidade = (
        matriz_similaridade.shape[0]
    )

    agrupador = Agrupador(
        quantidade
    )

    for primeiro in range(quantidade):
        for segundo in range(
            primeiro + 1,
            quantidade,
        ):
            distancia = (
                segundo - primeiro
            )

            if distancia == 1:
                limiar = (
                    LIMIAR_SIMILARIDADE_ADJACENTE
                )

            else:
                limiar = (
                    LIMIAR_SIMILARIDADE_DISTANTE
                )

            if (
                matriz_similaridade[
                    primeiro,
                    segundo,
                ]
                >= limiar
            ):
                agrupador.unir(
                    primeiro,
                    segundo,
                )

    raizes = [
        agrupador.encontrar(indice)
        for indice in range(quantidade)
    ]

    mapa_rotulos = {}
    proximo_rotulo = 0
    rotulos = []

    for raiz in raizes:
        if raiz not in mapa_rotulos:
            mapa_rotulos[raiz] = (
                numero_para_rotulo(
                    proximo_rotulo
                )
            )

            proximo_rotulo += 1

        rotulos.append(
            mapa_rotulos[raiz]
        )

    return rotulos


def criar_blocos_estrutura(
    linha_do_tempo: list[dict],
    rotulos: list[str],
) -> list[dict]:
    blocos = []

    for indice, segmento in enumerate(
        linha_do_tempo
    ):
        rotulo = rotulos[indice]

        if (
            blocos
            and blocos[-1]["rotulo"]
            == rotulo
        ):
            blocos[-1]["fim_segundos"] = (
                segmento["fim_segundos"]
            )

            blocos[-1]["fim_formatado"] = (
                segmento["fim_formatado"]
            )

            blocos[-1]["segmentos"].append(
                segmento["segmento"]
            )

            blocos[-1]["energias"].append(
                segmento["energia_relativa"]
            )

            blocos[-1]["brilhos"].append(
                segmento["brilho_relativo"]
            )

        else:
            blocos.append({
                "rotulo": rotulo,
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
                "energias": [
                    segmento["energia_relativa"]
                ],
                "brilhos": [
                    segmento["brilho_relativo"]
                ],
            })

    for bloco in blocos:
        bloco["energia_media"] = round(
            float(
                np.mean(
                    bloco.pop("energias")
                )
            ),
            4,
        )

        bloco["brilho_medio"] = round(
            float(
                np.mean(
                    bloco.pop("brilhos")
                )
            ),
            4,
        )

        bloco["duracao_segundos"] = round(
            bloco["fim_segundos"]
            - bloco["inicio_segundos"],
            2,
        )

    return blocos


def sugerir_funcoes_blocos(
    blocos: list[dict],
) -> list[dict]:
    ocorrencias = {}

    for bloco in blocos:
        rotulo = bloco["rotulo"]

        ocorrencias.setdefault(
            rotulo,
            [],
        ).append(bloco)

    quantidade_blocos = len(
        blocos
    )

    for indice, bloco in enumerate(
        blocos
    ):
        repeticoes = len(
            ocorrencias[
                bloco["rotulo"]
            ]
        )

        primeira_posicao = (
            indice == 0
        )

        ultima_posicao = (
            indice
            == quantidade_blocos - 1
        )

        if (
            primeira_posicao
            and repeticoes == 1
        ):
            hipotese = (
                "possível introdução"
            )

        elif (
            ultima_posicao
            and bloco["energia_media"]
            < 0.45
        ):
            hipotese = (
                "possível encerramento"
            )

        elif (
            repeticoes >= 2
            and bloco["energia_media"]
            >= 0.60
        ):
            hipotese = (
                "possível refrão "
                "ou seção de destaque"
            )

        elif repeticoes >= 2:
            hipotese = (
                "possível verso "
                "ou seção recorrente"
            )

        elif (
            not primeira_posicao
            and not ultima_posicao
        ):
            hipotese = (
                "possível ponte "
                "ou transição"
            )

        else:
            hipotese = "seção única"

        bloco["quantidade_ocorrencias"] = (
            repeticoes
        )

        bloco["hipotese_funcao"] = (
            hipotese
        )

    return blocos


def analisar_repeticoes(
    audio: np.ndarray,
    taxa_amostragem: int,
    dinamica: dict,
) -> dict:
    linha_do_tempo = dinamica[
        "linha_do_tempo"
    ]

    matriz_similaridade = (
        criar_matriz_similaridade(
            audio=audio,
            taxa_amostragem=taxa_amostragem,
            linha_do_tempo=linha_do_tempo,
        )
    )

    rotulos = agrupar_segmentos_similares(
        matriz_similaridade
    )

    blocos = criar_blocos_estrutura(
        linha_do_tempo,
        rotulos,
    )

    blocos = sugerir_funcoes_blocos(
        blocos
    )

    mapa_segmentos = " - ".join(
        rotulos
    )

    mapa_blocos = " - ".join(
        bloco["rotulo"]
        for bloco in blocos
    )

    rotulos_repetidos = sorted({
        bloco["rotulo"]
        for bloco in blocos
        if bloco[
            "quantidade_ocorrencias"
        ] >= 2
    })

    return {
        "limiar_similaridade_distante": (
            LIMIAR_SIMILARIDADE_DISTANTE
        ),
        "limiar_similaridade_adjacente": (
            LIMIAR_SIMILARIDADE_ADJACENTE
        ),
        "mapa_segmentos": mapa_segmentos,
        "mapa_blocos": mapa_blocos,
        "rotulos_segmentos": rotulos,
        "rotulos_repetidos": (
            rotulos_repetidos
        ),
        "quantidade_blocos": len(
            blocos
        ),
        "blocos": blocos,
        "matriz_similaridade": (
            matriz_similaridade
        ),
    }


def gerar_grafico_dinamica(
    dinamica: dict,
    transicoes: list[dict],
    caminho_grafico: Path,
    nome_musica: str,
) -> None:
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


def gerar_grafico_similaridade(
    matriz_similaridade: np.ndarray,
    caminho_grafico: Path,
    nome_musica: str,
) -> None:
    figura, eixo = plt.subplots(
        figsize=(9, 8)
    )

    imagem = eixo.imshow(
        matriz_similaridade,
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    eixo.set_title(
        f"Similaridade entre trechos: "
        f"{nome_musica}"
    )

    eixo.set_xlabel(
        "Segmento comparado"
    )

    eixo.set_ylabel(
        "Segmento de referência"
    )

    quantidade = (
        matriz_similaridade.shape[0]
    )

    marcacoes = np.arange(
        quantidade
    )

    eixo.set_xticks(
        marcacoes
    )

    eixo.set_yticks(
        marcacoes
    )

    eixo.set_xticklabels(
        marcacoes + 1,
        rotation=90,
        fontsize=7,
    )

    eixo.set_yticklabels(
        marcacoes + 1,
        fontsize=7,
    )

    figura.colorbar(
        imagem,
        ax=eixo,
        label="Similaridade",
    )

    figura.tight_layout()

    figura.savefig(
        caminho_grafico,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figura
    )


def analisar_audio(
    caminho_audio: Path,
) -> dict:
    print()
    print(
        f"Carregando: "
        f"{caminho_audio.name}"
    )

    audio, taxa_amostragem = librosa.load(
        caminho_audio,
        sr=None,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(
            "O arquivo foi carregado, "
            "mas não contém áudio."
        )

    basico = analisar_caracteristicas_basicas(
        audio=audio,
        taxa_amostragem=taxa_amostragem,
    )

    print(
        "Analisando dinâmica ao longo "
        "da música..."
    )

    dinamica = analisar_dinamica(
        audio=audio,
        taxa_amostragem=taxa_amostragem,
    )

    print(
        "Detectando mudanças entre "
        "os segmentos..."
    )

    transicoes = detectar_transicoes(
        dinamica
    )

    print(
        "Procurando trechos semelhantes "
        "e repetições..."
    )

    repeticoes = analisar_repeticoes(
        audio=audio,
        taxa_amostragem=taxa_amostragem,
        dinamica=dinamica,
    )

    matriz_similaridade = (
        repeticoes.pop(
            "matriz_similaridade"
        )
    )

    resultado = {
        "arquivo": caminho_audio.name,
        "formato": (
            caminho_audio.suffix
            .lower()
            .replace(".", "")
        ),
        **basico,
        "dinamica": dinamica,
        "estrutura_aproximada": {
            "quantidade_transicoes": len(
                transicoes
            ),
            "transicoes": transicoes,
            "repeticoes": repeticoes,
        },
    }

    return {
        "resultado": resultado,
        "matriz_similaridade": (
            matriz_similaridade
        ),
    }


def criar_descricao(
    resultado: dict,
) -> str:
    ritmo = resultado[
        "ritmo"
    ]

    tonalidade = resultado[
        "tonalidade"
    ]

    energia = resultado[
        "energia"
    ]

    espectro = resultado[
        "espectro"
    ]

    duracao = resultado[
        "duracao"
    ]

    dinamica = resultado[
        "dinamica"
    ]

    estrutura = resultado[
        "estrutura_aproximada"
    ]

    repeticoes = estrutura[
        "repeticoes"
    ]

    picos = dinamica[
        "picos_principais"
    ]

    if picos:
        primeiro_pico = picos[0]

        descricao_pico = (
            f"O principal pico de intensidade "
            f"aparece aproximadamente entre "
            f"{primeiro_pico['inicio']} e "
            f"{primeiro_pico['fim']}."
        )

    else:
        descricao_pico = (
            "Não foi possível identificar "
            "um pico principal."
        )

    return (
        f"Faixa com aproximadamente "
        f"{duracao['minutos']:.2f} minutos, "
        f"andamento {ritmo['classificacao']} "
        f"em cerca de "
        f"{ritmo['bpm_estimado']:.0f} BPM. "
        f"A tonalidade provável é "
        f"{tonalidade['tonalidade_completa']}. "
        f"A energia geral foi classificada como "
        f"{energia['classificacao']}, "
        f"com perfil sonoro "
        f"{espectro['brilho_estimado']}. "
        f"A variação dinâmica é "
        f"{dinamica['variacao_dinamica']} "
        f"e a faixa "
        f"{dinamica['tendencia_geral']}. "
        f"Foram detectadas "
        f"{estrutura['quantidade_transicoes']} "
        f"mudanças relevantes. "
        f"O mapa estrutural aproximado é "
        f"{repeticoes['mapa_blocos']}. "
        f"{descricao_pico}"
    )


def salvar_csv_dinamica(
    dinamica: dict,
    caminho_csv: Path,
) -> None:
    campos = [
        "segmento",
        "inicio_segundos",
        "fim_segundos",
        "inicio_formatado",
        "fim_formatado",
        "energia_media_rms",
        "energia_maxima_rms",
        "energia_relativa",
        "brilho_medio_hz",
        "brilho_relativo",
        "classificacao",
    ]

    with caminho_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo_csv:
        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=campos,
            delimiter=";",
        )

        escritor.writeheader()

        escritor.writerows(
            dinamica["linha_do_tempo"]
        )


def salvar_csv_transicoes(
    transicoes: list[dict],
    caminho_csv: Path,
) -> None:
    campos = [
        "tempo_segundos",
        "tempo_formatado",
        "segmento_anterior",
        "segmento_atual",
        "evento",
        "variacao_energia",
        "variacao_brilho",
        "pontuacao_mudanca",
        "mudou_classificacao",
        "energia_anterior",
        "energia_atual",
        "brilho_anterior",
        "brilho_atual",
    ]

    with caminho_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo_csv:
        escritor = csv.DictWriter(
            arquivo_csv,
            fieldnames=campos,
            delimiter=";",
        )

        escritor.writeheader()

        escritor.writerows(
            transicoes
        )


def salvar_csv_similaridade(
    matriz: np.ndarray,
    caminho_csv: Path,
) -> None:
    with caminho_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo_csv:
        escritor = csv.writer(
            arquivo_csv,
            delimiter=";",
        )

        quantidade = matriz.shape[0]

        cabecalho = [
            "segmento"
        ] + [
            str(indice + 1)
            for indice in range(
                quantidade
            )
        ]

        escritor.writerow(
            cabecalho
        )

        for indice in range(
            quantidade
        ):
            linha = [
                indice + 1
            ] + [
                round(
                    float(valor),
                    4,
                )
                for valor
                in matriz[indice]
            ]

            escritor.writerow(
                linha
            )


def criar_texto_mapa_estrutura(
    repeticoes: dict,
) -> str:
    linhas = [
        "MAPA ESTRUTURAL APROXIMADO",
        "==========================",
        "",
        (
            f"Segmentos: "
            f"{repeticoes['mapa_segmentos']}"
        ),
        (
            f"Blocos: "
            f"{repeticoes['mapa_blocos']}"
        ),
        "",
        "BLOCOS DETECTADOS",
        "-----------------",
    ]

    for bloco in repeticoes[
        "blocos"
    ]:
        linhas.append(
            f"{bloco['rotulo']}: "
            f"{bloco['inicio_formatado']} até "
            f"{bloco['fim_formatado']} | "
            f"{bloco['hipotese_funcao']} | "
            f"energia "
            f"{bloco['energia_media']:.2f} | "
            f"{bloco['quantidade_ocorrencias']} "
            f"ocorrência(s)"
        )

    linhas.extend([
        "",
        "Observação:",
        (
            "Os nomes das funções são hipóteses "
            "baseadas em repetição e energia. "
            "Não são identificações musicais "
            "definitivas."
        ),
    ])

    return "\n".join(
        linhas
    )


def salvar_resultados(
    resultado: dict,
    descricao: str,
    matriz_similaridade: np.ndarray,
    pasta_output: Path,
    caminho_audio: Path,
) -> Path:
    nome_base = normalizar_nome_arquivo(
        caminho_audio.stem
    )

    pasta_musica = (
        pasta_output / nome_base
    )

    contador = 2

    while pasta_musica.exists():
        pasta_musica = (
            pasta_output
            / f"{nome_base}_{contador}"
        )

        contador += 1

    pasta_musica.mkdir(
        parents=True,
        exist_ok=False,
    )

    caminho_json = (
        pasta_musica
        / "analise.json"
    )

    caminho_descricao = (
        pasta_musica
        / "descricao.txt"
    )

    caminho_dinamica = (
        pasta_musica
        / "dinamica.csv"
    )

    caminho_transicoes = (
        pasta_musica
        / "transicoes.csv"
    )

    caminho_similaridade = (
        pasta_musica
        / "similaridade.csv"
    )

    caminho_mapa = (
        pasta_musica
        / "mapa_estrutura.txt"
    )

    caminho_grafico_dinamica = (
        pasta_musica
        / "grafico_dinamica.png"
    )

    caminho_grafico_similaridade = (
        pasta_musica
        / "grafico_similaridade.png"
    )

    with caminho_json.open(
        "w",
        encoding="utf-8",
    ) as arquivo_json:
        json.dump(
            resultado,
            arquivo_json,
            ensure_ascii=False,
            indent=4,
        )

    with caminho_descricao.open(
        "w",
        encoding="utf-8",
    ) as arquivo_txt:
        arquivo_txt.write(
            descricao
        )

    repeticoes = resultado[
        "estrutura_aproximada"
    ]["repeticoes"]

    texto_mapa = (
        criar_texto_mapa_estrutura(
            repeticoes
        )
    )

    with caminho_mapa.open(
        "w",
        encoding="utf-8",
    ) as arquivo_mapa:
        arquivo_mapa.write(
            texto_mapa
        )

    salvar_csv_dinamica(
        resultado["dinamica"],
        caminho_dinamica,
    )

    salvar_csv_transicoes(
        resultado[
            "estrutura_aproximada"
        ]["transicoes"],
        caminho_transicoes,
    )

    salvar_csv_similaridade(
        matriz_similaridade,
        caminho_similaridade,
    )

    print(
        "Gerando gráfico da dinâmica..."
    )

    gerar_grafico_dinamica(
        dinamica=resultado[
            "dinamica"
        ],
        transicoes=resultado[
            "estrutura_aproximada"
        ]["transicoes"],
        caminho_grafico=(
            caminho_grafico_dinamica
        ),
        nome_musica=(
            caminho_audio.stem
        ),
    )

    print(
        "Gerando mapa visual "
        "de similaridade..."
    )

    gerar_grafico_similaridade(
        matriz_similaridade=(
            matriz_similaridade
        ),
        caminho_grafico=(
            caminho_grafico_similaridade
        ),
        nome_musica=(
            caminho_audio.stem
        ),
    )

    print()
    print(
        f"Resultados salvos em: "
        f"{pasta_musica}"
    )

    print("Arquivos gerados:")
    print("- analise.json")
    print("- descricao.txt")
    print("- dinamica.csv")
    print("- transicoes.csv")
    print("- similaridade.csv")
    print("- mapa_estrutura.txt")
    print("- grafico_dinamica.png")
    print("- grafico_similaridade.png")

    return pasta_musica


def exibir_resumo(
    resultado: dict,
) -> None:
    dinamica = resultado[
        "dinamica"
    ]

    estrutura = resultado[
        "estrutura_aproximada"
    ]

    repeticoes = estrutura[
        "repeticoes"
    ]

    print()
    print("DINÂMICA")
    print("--------")

    print(
        f"Variação dinâmica: "
        f"{dinamica['variacao_dinamica']}"
    )

    print(
        f"Tendência: "
        f"{dinamica['tendencia_geral']}"
    )

    print()
    print("MUDANÇAS RELEVANTES")
    print("-------------------")

    transicoes = estrutura[
        "transicoes"
    ]

    if not transicoes:
        print(
            "Nenhuma mudança forte "
            "foi detectada."
        )

    else:
        for transicao in transicoes:
            print(
                f"{transicao['tempo_formatado']} "
                f"{transicao['evento']}"
            )

    print()
    print("ESTRUTURA APROXIMADA")
    print("--------------------")

    print(
        f"Mapa por segmentos: "
        f"{repeticoes['mapa_segmentos']}"
    )

    print(
        f"Mapa por blocos: "
        f"{repeticoes['mapa_blocos']}"
    )

    print()
    print("Blocos:")

    for bloco in repeticoes[
        "blocos"
    ]:
        print(
            f"{bloco['rotulo']} | "
            f"{bloco['inicio_formatado']} até "
            f"{bloco['fim_formatado']} | "
            f"{bloco['hipotese_funcao']}"
        )


def main() -> None:
    pasta_projeto = (
        Path(__file__).resolve().parent
    )

    pasta_input = (
        pasta_projeto / "input"
    )

    pasta_output = (
        pasta_projeto / "output"
    )

    pasta_input.mkdir(
        exist_ok=True
    )

    pasta_output.mkdir(
        exist_ok=True
    )

    print()
    print("Music Prompt Analyzer")
    print("=====================")
    print()

    try:
        arquivos = listar_arquivos_audio(
            pasta_input
        )

        caminho_audio = escolher_arquivo_audio(
            arquivos
        )

        analise_completa = analisar_audio(
            caminho_audio
        )

        resultado = analise_completa[
            "resultado"
        ]

        matriz_similaridade = (
            analise_completa[
                "matriz_similaridade"
            ]
        )

        descricao = criar_descricao(
            resultado
        )

        pasta_resultado = salvar_resultados(
            resultado=resultado,
            descricao=descricao,
            matriz_similaridade=(
                matriz_similaridade
            ),
            pasta_output=pasta_output,
            caminho_audio=caminho_audio,
        )

        print()
        print("RESUMO")
        print("------")
        print(descricao)

        exibir_resumo(
            resultado
        )

        print()
        print(
            "Análise concluída em: "
            f"{pasta_resultado}"
        )

    except FileNotFoundError as erro:
        print()
        print("ERRO")
        print("----")
        print(erro)

    except KeyboardInterrupt:
        print()
        print()
        print(
            "Operação cancelada "
            "pelo usuário."
        )

    except Exception as erro:
        print()
        print(
            "Ocorreu um erro durante "
            "a análise:"
        )

        print(
            f"{type(erro).__name__}: "
            f"{erro}"
        )


if __name__ == "__main__":
    main()