from pathlib import Path

from src.audio_files import (
    escolher_arquivo_audio,
    listar_arquivos_audio,
)
from src.pipeline import (
    analisar_audio,
    criar_descricao,
)
from src.reporting import (
    exibir_resumo,
    salvar_resultados,
)


def main() -> None:
    """
    Ponto de entrada do Music Prompt Analyzer.
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

        exibir_resumo(
            resultado=resultado,
            descricao=descricao,
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