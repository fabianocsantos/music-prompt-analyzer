import csv
import json
from pathlib import Path

import numpy as np

from src.component_analysis import (
    gerar_grafico_componentes,
)
from src.dynamics import (
    gerar_grafico_dinamica,
)
from src.helpers import (
    normalizar_nome_arquivo,
)
from src.structure_analysis import (
    gerar_grafico_similaridade,
)


def salvar_csv_dinamica(
    dinamica: dict,
    caminho_csv: Path,
) -> None:
    """
    Salva a linha do tempo dinâmica em CSV.
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


def salvar_csv_componentes(
    componentes: dict,
    caminho_csv: Path,
) -> None:
    """
    Salva a evolução harmônica e percussiva em CSV.
    """

    campos = [
        "segmento",
        "inicio_segundos",
        "fim_segundos",
        "inicio_formatado",
        "fim_formatado",
        "energia_harmonica",
        "energia_percussiva",
        "proporcao_harmonica",
        "proporcao_percussiva",
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
            componentes["linha_do_tempo"]
        )


def salvar_csv_transicoes(
    transicoes: list[dict],
    caminho_csv: Path,
) -> None:
    """
    Salva as transições detectadas em CSV.
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
    Salva a matriz de similaridade em CSV.
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
                for valor in matriz[indice]
            ]

            escritor.writerow(
                linha
            )


def criar_texto_mapa_estrutura(
    repeticoes: dict,
) -> str:
    """
    Cria o relatório textual da estrutura aproximada.
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


def criar_pasta_resultado(
    pasta_output: Path,
    caminho_audio: Path,
) -> Path:
    """
    Cria uma pasta exclusiva para cada análise.
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

    return pasta_musica


def salvar_resultados(
    resultado: dict,
    descricao: str,
    matriz_similaridade: np.ndarray,
    pasta_output: Path,
    caminho_audio: Path,
) -> Path:
    """
    Salva todos os resultados e gráficos da análise.
    """

    pasta_musica = criar_pasta_resultado(
        pasta_output=pasta_output,
        caminho_audio=caminho_audio,
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

    caminho_componentes = (
        pasta_musica
        / "componentes.csv"
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

    caminho_grafico_componentes = (
        pasta_musica
        / "grafico_componentes.png"
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

    texto_mapa = criar_texto_mapa_estrutura(
        repeticoes
    )

    with caminho_mapa.open(
        "w",
        encoding="utf-8",
    ) as arquivo_mapa:
        arquivo_mapa.write(
            texto_mapa
        )

    salvar_csv_dinamica(
        dinamica=resultado[
            "dinamica"
        ],
        caminho_csv=caminho_dinamica,
    )

    salvar_csv_componentes(
        componentes=resultado[
            "componentes"
        ],
        caminho_csv=caminho_componentes,
    )

    salvar_csv_transicoes(
        transicoes=resultado[
            "estrutura_aproximada"
        ]["transicoes"],
        caminho_csv=caminho_transicoes,
    )

    salvar_csv_similaridade(
        matriz=matriz_similaridade,
        caminho_csv=caminho_similaridade,
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
        "Gerando gráfico de componentes..."
    )

    gerar_grafico_componentes(
        componentes=resultado[
            "componentes"
        ],
        caminho_grafico=(
            caminho_grafico_componentes
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
    print("- componentes.csv")
    print("- transicoes.csv")
    print("- similaridade.csv")
    print("- mapa_estrutura.txt")
    print("- grafico_dinamica.png")
    print("- grafico_componentes.png")
    print("- grafico_similaridade.png")

    return pasta_musica


def exibir_resumo(
    resultado: dict,
    descricao: str,
) -> None:
    """
    Exibe o resumo completo no terminal.
    """

    componentes = resultado[
        "componentes"
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

    print()
    print("RESUMO")
    print("------")
    print(descricao)

    print()
    print("COMPONENTES")
    print("-----------")

    print(
        f"Equilíbrio geral: "
        f"{componentes['classificacao']}"
    )

    print(
        f"Conteúdo harmônico: "
        f"{componentes['percentual_harmonico']:.2f}%"
    )

    print(
        f"Conteúdo percussivo: "
        f"{componentes['percentual_percussivo']:.2f}%"
    )

    destaque_harmonico = componentes[
        "destaques"
    ]["mais_harmonico"]

    destaque_percussivo = componentes[
        "destaques"
    ]["mais_percussivo"]

    if destaque_harmonico:
        print(
            f"Trecho mais harmônico: "
            f"{destaque_harmonico['inicio']} até "
            f"{destaque_harmonico['fim']}"
        )

    if destaque_percussivo:
        print(
            f"Trecho mais percussivo: "
            f"{destaque_percussivo['inicio']} até "
            f"{destaque_percussivo['fim']}"
        )

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