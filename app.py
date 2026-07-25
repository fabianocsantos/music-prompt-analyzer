import csv
import json
from pathlib import Path

import librosa
import numpy as np

from src.audio_files import (
    escolher_arquivo_audio,
    listar_arquivos_audio,
)
from src.basic_analysis import (
    analisar_caracteristicas_basicas,
)
from src.dynamics import (
    analisar_dinamica,
    detectar_transicoes,
    gerar_grafico_dinamica,
)
from src.helpers import (
    normalizar_nome_arquivo,
)
from src.structure_analysis import (
    analisar_repeticoes,
    gerar_grafico_similaridade,
)


def analisar_audio(
    caminho_audio: Path,
) -> dict:
    """
    Executa todas as etapas de análise musical.
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

    matriz_similaridade = repeticoes.pop(
        "matriz_similaridade"
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
    Cria uma descrição textual resumida da música.
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


def salvar_resultados(
    resultado: dict,
    descricao: str,
    matriz_similaridade: np.ndarray,
    pasta_output: Path,
    caminho_audio: Path,
) -> Path:
    """
    Salva todos os resultados da análise.
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
    Exibe um resumo no terminal.
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