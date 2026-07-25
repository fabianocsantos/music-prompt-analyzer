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
    LIMIAR_SIMILARIDADE_ADJACENTE,
    LIMIAR_SIMILARIDADE_DISTANTE,
)
from src.dynamics import (
    analisar_dinamica,
    detectar_transicoes,
    gerar_grafico_dinamica,
)
from src.helpers import (
    normalizar_nome_arquivo,
    numero_para_rotulo,
)


def extrair_vetor_segmento(
    trecho_audio: np.ndarray,
    taxa_amostragem: int,
) -> np.ndarray:
    """
    Extrai características acústicas de um segmento.
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
    Padroniza e normaliza os vetores acústicos.
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

    return padronizados / normas


def criar_matriz_similaridade(
    audio: np.ndarray,
    taxa_amostragem: int,
    linha_do_tempo: list[dict],
) -> np.ndarray:
    """
    Compara todos os segmentos entre si.
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
    """
    Estrutura auxiliar para unir segmentos semelhantes.
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
    Agrupa segmentos com similaridade acima dos limites definidos.
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
    """
    Sugere funções musicais aproximadas para os blocos.
    """

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


def analisar_audio(
    caminho_audio: Path,
) -> dict:
    """
    Executa todas as etapas de análise.
    """

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
    """
    Cria uma descrição textual resumida.
    """

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
    """
    Salva a linha do tempo dinâmica.
    """

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
    """
    Salva as transições detectadas.
    """

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
    """
    Salva a matriz de similaridade.
    """

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
    """
    Cria o relatório textual da estrutura.
    """

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
    """
    Salva todos os arquivos da análise.
    """

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
    """
    Exibe o resumo no terminal.
    """

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
    """
    Ponto de entrada do programa.
    """

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