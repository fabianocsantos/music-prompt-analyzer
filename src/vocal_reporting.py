import csv
from pathlib import Path

from src.vocal_analysis import (
    gerar_grafico_presenca_vocal,
)


def salvar_csv_presenca_vocal(
    analise_vocal: dict,
    caminho_csv: Path,
) -> None:
    """
    Salva a linha do tempo da estimativa vocal.
    """

    campos = [
        "segmento",
        "inicio_segundos",
        "fim_segundos",
        "inicio_formatado",
        "fim_formatado",
        "energia",
        "centroide_hz",
        "largura_hz",
        "taxa_cruzamento_zero",
        "planicidade_espectral",
        "proporcao_harmonica",
        "probabilidade_vocal",
        "classificacao",
        "pontuacao_energia",
        "pontuacao_harmonica",
        "pontuacao_tonalidade",
        "pontuacao_centroide",
        "pontuacao_cruzamento",
        "pontuacao_largura",
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
            analise_vocal["linha_do_tempo"]
        )


def criar_relatorio_vocal(
    analise_vocal: dict,
) -> str:
    """
    Cria um resumo textual da estimativa vocal.
    """

    linhas = [
        "PRESENÇA VOCAL PROVÁVEL",
        "=======================",
        "",
        (
            f"Classificação geral: "
            f"{analise_vocal['classificacao_geral']}"
        ),
        (
            f"Probabilidade média: "
            f"{analise_vocal['probabilidade_media']:.2f}"
        ),
        (
            f"Segmentos com probabilidade alta: "
            f"{analise_vocal['segmentos_alta_probabilidade']}"
        ),
        (
            f"Segmentos com probabilidade média: "
            f"{analise_vocal['segmentos_media_probabilidade']}"
        ),
        (
            f"Segmentos com probabilidade baixa: "
            f"{analise_vocal['segmentos_baixa_probabilidade']}"
        ),
        "",
        "REGIÕES PROVÁVEIS",
        "-----------------",
    ]

    regioes = analise_vocal[
        "regioes_provaveis"
    ]

    if not regioes:
        linhas.append(
            "Nenhuma região vocal provável foi destacada."
        )

    else:
        for indice, regiao in enumerate(
            regioes,
            start=1,
        ):
            linhas.append(
                f"{indice}. "
                f"{regiao['inicio_formatado']} até "
                f"{regiao['fim_formatado']} | "
                f"{regiao['classificacao']} | "
                f"pontuação "
                f"{regiao['pontuacao_media']:.2f}"
            )

    linhas.extend([
        "",
        "Observação:",
        (
            "Esta análise utiliza uma heurística acústica. "
            "Instrumentos melódicos podem ser confundidos "
            "com voz. O método não separa nem identifica "
            "cantores."
        ),
    ])

    return "\n".join(
        linhas
    )


def salvar_relatorios_vocais(
    pasta_resultado: Path,
    resultado: dict,
    nome_musica: str,
) -> None:
    """
    Salva CSV, TXT e gráfico da presença vocal provável.
    """

    analise_vocal = resultado[
        "presenca_vocal_provavel"
    ]

    caminho_csv = (
        pasta_resultado
        / "voz_provavel.csv"
    )

    caminho_txt = (
        pasta_resultado
        / "voz_provavel.txt"
    )

    caminho_grafico = (
        pasta_resultado
        / "grafico_voz_provavel.png"
    )

    salvar_csv_presenca_vocal(
        analise_vocal=analise_vocal,
        caminho_csv=caminho_csv,
    )

    relatorio = criar_relatorio_vocal(
        analise_vocal
    )

    with caminho_txt.open(
        "w",
        encoding="utf-8",
    ) as arquivo_txt:
        arquivo_txt.write(
            relatorio
        )

    print(
        "Gerando gráfico de presença vocal provável..."
    )

    gerar_grafico_presenca_vocal(
        analise_vocal=analise_vocal,
        caminho_grafico=caminho_grafico,
        nome_musica=nome_musica,
    )

    print("Arquivos vocais gerados:")
    print("- voz_provavel.csv")
    print("- voz_provavel.txt")
    print("- grafico_voz_provavel.png")