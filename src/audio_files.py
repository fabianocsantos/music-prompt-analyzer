from pathlib import Path

from src.config import EXTENSOES_SUPORTADAS


def listar_arquivos_audio(
    pasta_input: Path,
) -> list[Path]:
    """
    Retorna os arquivos de áudio suportados da pasta de entrada.
    """

    arquivos = [
        arquivo
        for arquivo in pasta_input.iterdir()
        if arquivo.is_file()
        and arquivo.suffix.lower()
        in EXTENSOES_SUPORTADAS
    ]

    arquivos.sort(
        key=lambda arquivo: arquivo.name.lower()
    )

    return arquivos


def escolher_arquivo_audio(
    arquivos: list[Path],
) -> Path:
    """
    Exibe os arquivos encontrados e permite escolher um pelo número.
    """

    if not arquivos:
        extensoes = ", ".join(
            sorted(EXTENSOES_SUPORTADAS)
        )

        raise FileNotFoundError(
            "Nenhum arquivo de áudio foi encontrado "
            "na pasta input.\n"
            f"Formatos aceitos: {extensoes}"
        )

    if len(arquivos) == 1:
        arquivo = arquivos[0]

        print("Uma música foi encontrada:")
        print(f"1. {arquivo.name}")
        print()
        print(
            f"Selecionando automaticamente: "
            f"{arquivo.name}"
        )

        return arquivo

    print("Músicas encontradas:")
    print()

    for indice, arquivo in enumerate(
        arquivos,
        start=1,
    ):
        tamanho_mb = (
            arquivo.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{indice}. {arquivo.name} "
            f"({tamanho_mb:.2f} MB)"
        )

    print()

    while True:
        resposta = input(
            "Digite o número da música que deseja "
            f"analisar [1-{len(arquivos)}]: "
        ).strip()

        try:
            numero_escolhido = int(resposta)

        except ValueError:
            print(
                "Digite apenas o número correspondente "
                "à música."
            )
            continue

        if 1 <= numero_escolhido <= len(arquivos):
            return arquivos[numero_escolhido - 1]

        print(
            f"Escolha um número entre 1 "
            f"e {len(arquivos)}."
        )