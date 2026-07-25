from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    LIMIAR_SIMILARIDADE_ADJACENTE,
    LIMIAR_SIMILARIDADE_DISTANTE,
)
from src.helpers import numero_para_rotulo


def extrair_vetor_segmento(
    trecho_audio: np.ndarray,
    taxa_amostragem: int,
) -> np.ndarray:
    """
    Extrai um conjunto de características acústicas de um trecho.

    O vetor combina:
    - cromas;
    - MFCCs;
    - contraste espectral;
    - energia;
    - brilho;
    - largura espectral;
    - taxa de cruzamento por zero.
    """

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
    """
    Padroniza os vetores e normaliza seu comprimento.

    Isso permite comparar características com escalas diferentes.
    """

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

    return (
        padronizados
        / normas
    )


def criar_matriz_similaridade(
    audio: np.ndarray,
    taxa_amostragem: int,
    linha_do_tempo: list[dict],
) -> np.ndarray:
    """
    Compara todos os segmentos da música entre si.
    """

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
            trecho_audio=trecho_audio,
            taxa_amostragem=taxa_amostragem,
        )

        vetores.append(
            vetor
        )

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
    """
    Estrutura auxiliar para reunir segmentos semelhantes.

    Usa uma técnica conhecida como union-find.
    """

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
    """
    Agrupa segmentos que ultrapassam os limites de similaridade.
    """

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

            similaridade = matriz_similaridade[
                primeiro,
                segundo,
            ]

            if similaridade >= limiar:
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
    """
    Junta segmentos consecutivos com o mesmo rótulo.
    """

    blocos = []

    for indice, segmento in enumerate(
        linha_do_tempo
    ):
        rotulo = rotulos[indice]

        if (
            blocos
            and blocos[-1]["rotulo"] == rotulo
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
        energias = bloco.pop(
            "energias"
        )

        brilhos = bloco.pop(
            "brilhos"
        )

        bloco["energia_media"] = round(
            float(
                np.mean(energias)
            ),
            4,
        )

        bloco["brilho_medio"] = round(
            float(
                np.mean(brilhos)
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
    """
    Sugere possíveis funções musicais para cada bloco.

    As sugestões são hipóteses baseadas em:
    - posição;
    - repetição;
    - energia.
    """

    ocorrencias = {}

    for bloco in blocos:
        rotulo = bloco["rotulo"]

        ocorrencias.setdefault(
            rotulo,
            [],
        ).append(
            bloco
        )

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
            and bloco["energia_media"] < 0.45
        ):
            hipotese = (
                "possível encerramento"
            )

        elif (
            repeticoes >= 2
            and bloco["energia_media"] >= 0.60
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
            hipotese = (
                "seção única"
            )

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
    """
    Cria a matriz de similaridade e o mapa estrutural.
    """

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
        linha_do_tempo=linha_do_tempo,
        rotulos=rotulos,
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


def gerar_grafico_similaridade(
    matriz_similaridade: np.ndarray,
    caminho_grafico: Path,
    nome_musica: str,
) -> None:
    """
    Gera uma imagem da matriz de similaridade.
    """

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