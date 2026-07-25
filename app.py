import json
import re
from pathlib import Path

import librosa
import numpy as np


EXTENSOES_SUPORTADAS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".ogg",
}


NOMES_NOTAS = [
    "Dó",
    "Dó sustenido",
    "Ré",
    "Ré sustenido",
    "Mi",
    "Fá",
    "Fá sustenido",
    "Sol",
    "Sol sustenido",
    "Lá",
    "Lá sustenido",
    "Si",
]


PERFIL_MAIOR = np.array([
    6.35,
    2.23,
    3.48,
    2.33,
    4.38,
    4.09,
    2.52,
    5.19,
    2.39,
    3.66,
    2.29,
    2.88,
])


PERFIL_MENOR = np.array([
    6.33,
    2.68,
    3.52,
    5.38,
    2.60,
    3.53,
    2.54,
    4.75,
    3.98,
    2.69,
    3.34,
    3.17,
])


def listar_arquivos_audio(pasta_input: Path) -> list[Path]:
    """
    Retorna todos os arquivos de áudio suportados da pasta input.
    """

    arquivos = [
        arquivo
        for arquivo in pasta_input.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_SUPORTADAS
    ]

    arquivos.sort(key=lambda arquivo: arquivo.name.lower())

    return arquivos


def escolher_arquivo_audio(arquivos: list[Path]) -> Path:
    """
    Exibe os arquivos encontrados e permite escolher um pelo número.
    """

    if not arquivos:
        extensoes = ", ".join(sorted(EXTENSOES_SUPORTADAS))

        raise FileNotFoundError(
            "Nenhum arquivo de áudio foi encontrado na pasta input.\n"
            f"Formatos aceitos: {extensoes}"
        )

    if len(arquivos) == 1:
        arquivo = arquivos[0]

        print("Uma música foi encontrada:")
        print(f"1. {arquivo.name}")
        print()
        print(f"Selecionando automaticamente: {arquivo.name}")

        return arquivo

    print("Músicas encontradas:")
    print()

    for indice, arquivo in enumerate(arquivos, start=1):
        tamanho_mb = arquivo.stat().st_size / (1024 * 1024)

        print(
            f"{indice}. {arquivo.name} "
            f"({tamanho_mb:.2f} MB)"
        )

    print()

    while True:
        resposta = input(
            f"Digite o número da música que deseja analisar "
            f"[1-{len(arquivos)}]: "
        ).strip()

        try:
            numero_escolhido = int(resposta)
        except ValueError:
            print("Digite apenas o número correspondente à música.")
            continue

        if 1 <= numero_escolhido <= len(arquivos):
            return arquivos[numero_escolhido - 1]

        print(
            f"Escolha um número entre 1 e {len(arquivos)}."
        )


def converter_para_numero(valor) -> float:
    """
    Converte valores do NumPy ou arrays com um elemento para float.
    """

    array = np.asarray(valor)

    if array.size == 0:
        return 0.0

    return float(array.reshape(-1)[0])


def classificar_andamento(bpm: float) -> str:
    """
    Gera uma descrição simples do andamento.
    """

    if bpm < 60:
        return "muito lento"

    if bpm < 80:
        return "lento"

    if bpm < 105:
        return "moderado"

    if bpm < 130:
        return "animado"

    if bpm < 160:
        return "rápido"

    return "muito rápido"


def classificar_energia(rms_medio: float) -> str:
    """
    Classifica aproximadamente a energia média do áudio.
    """

    if rms_medio < 0.03:
        return "muito baixa"

    if rms_medio < 0.07:
        return "baixa"

    if rms_medio < 0.14:
        return "média"

    if rms_medio < 0.22:
        return "alta"

    return "muito alta"


def classificar_brilho(centroide_medio: float) -> str:
    """
    Usa o centroide espectral para estimar o brilho do áudio.
    """

    if centroide_medio < 1500:
        return "escuro e encorpado"

    if centroide_medio < 2500:
        return "equilibrado"

    if centroide_medio < 4000:
        return "brilhante"

    return "muito brilhante"


def calcular_correlacao(
    perfil_audio: np.ndarray,
    perfil_tonal: np.ndarray,
) -> float:
    """
    Calcula a semelhança entre dois perfis tonais.
    """

    if np.std(perfil_audio) == 0:
        return 0.0

    if np.std(perfil_tonal) == 0:
        return 0.0

    correlacao = np.corrcoef(
        perfil_audio,
        perfil_tonal,
    )[0, 1]

    return float(correlacao)


def estimar_tonalidade(
    audio_harmonico: np.ndarray,
    taxa_amostragem: int,
) -> dict:
    """
    Estima a nota principal e o modo maior ou menor.
    """

    chroma = librosa.feature.chroma_cqt(
        y=audio_harmonico,
        sr=taxa_amostragem,
    )

    perfil_audio = np.mean(chroma, axis=1)

    melhor_correlacao = -1.0
    melhor_indice_nota = 0
    melhor_modo = "maior"

    for indice_nota in range(12):
        perfil_maior_rotacionado = np.roll(
            PERFIL_MAIOR,
            indice_nota,
        )

        perfil_menor_rotacionado = np.roll(
            PERFIL_MENOR,
            indice_nota,
        )

        correlacao_maior = calcular_correlacao(
            perfil_audio,
            perfil_maior_rotacionado,
        )

        correlacao_menor = calcular_correlacao(
            perfil_audio,
            perfil_menor_rotacionado,
        )

        if correlacao_maior > melhor_correlacao:
            melhor_correlacao = correlacao_maior
            melhor_indice_nota = indice_nota
            melhor_modo = "maior"

        if correlacao_menor > melhor_correlacao:
            melhor_correlacao = correlacao_menor
            melhor_indice_nota = indice_nota
            melhor_modo = "menor"

    nome_nota = NOMES_NOTAS[melhor_indice_nota]

    confianca_aproximada = max(
        0.0,
        min(
            1.0,
            (melhor_correlacao + 1.0) / 2.0,
        ),
    )

    return {
        "nota": nome_nota,
        "modo": melhor_modo,
        "tonalidade_completa": f"{nome_nota} {melhor_modo}",
        "correlacao": round(melhor_correlacao, 4),
        "confianca_aproximada": round(
            confianca_aproximada,
            4,
        ),
    }


def analisar_audio(caminho_audio: Path) -> dict:
    """
    Carrega o áudio e extrai características musicais básicas.
    """

    print()
    print(f"Carregando: {caminho_audio.name}")

    audio, taxa_amostragem = librosa.load(
        caminho_audio,
        sr=None,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(
            "O arquivo foi carregado, mas não contém áudio."
        )

    print("Calculando duração...")

    duracao_segundos = librosa.get_duration(
        y=audio,
        sr=taxa_amostragem,
    )

    print(
        "Separando componentes harmônicos "
        "e percussivos..."
    )

    audio_harmonico, audio_percussivo = (
        librosa.effects.hpss(audio)
    )

    print("Estimando BPM...")

    tempo, batidas = librosa.beat.beat_track(
        y=audio_percussivo,
        sr=taxa_amostragem,
    )

    bpm = converter_para_numero(tempo)

    print("Estimando tonalidade...")

    tonalidade = estimar_tonalidade(
        audio_harmonico=audio_harmonico,
        taxa_amostragem=taxa_amostragem,
    )

    print("Calculando energia...")

    rms = librosa.feature.rms(y=audio)

    rms_medio = float(np.mean(rms))
    rms_maximo = float(np.max(rms))

    print("Analisando características espectrais...")

    centroide = librosa.feature.spectral_centroid(
        y=audio,
        sr=taxa_amostragem,
    )

    largura_espectral = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=taxa_amostragem,
    )

    zero_crossing = librosa.feature.zero_crossing_rate(
        audio
    )

    centroide_medio = float(np.mean(centroide))
    largura_media = float(np.mean(largura_espectral))
    zero_crossing_medio = float(
        np.mean(zero_crossing)
    )

    resultado = {
        "arquivo": caminho_audio.name,
        "formato": caminho_audio.suffix.lower().replace(
            ".",
            "",
        ),
        "duracao": {
            "segundos": round(
                float(duracao_segundos),
                2,
            ),
            "minutos": round(
                float(duracao_segundos) / 60,
                2,
            ),
        },
        "audio": {
            "taxa_amostragem_hz": int(
                taxa_amostragem
            ),
            "canais_analisados": 1,
        },
        "ritmo": {
            "bpm_estimado": round(bpm, 2),
            "classificacao": classificar_andamento(
                bpm
            ),
            "batidas_detectadas": int(
                len(batidas)
            ),
        },
        "tonalidade": tonalidade,
        "energia": {
            "rms_medio": round(rms_medio, 6),
            "rms_maximo": round(rms_maximo, 6),
            "classificacao": classificar_energia(
                rms_medio
            ),
        },
        "espectro": {
            "centroide_medio_hz": round(
                centroide_medio,
                2,
            ),
            "largura_media_hz": round(
                largura_media,
                2,
            ),
            "taxa_cruzamento_zero": round(
                zero_crossing_medio,
                6,
            ),
            "brilho_estimado": classificar_brilho(
                centroide_medio
            ),
        },
    }

    return resultado


def criar_descricao(resultado: dict) -> str:
    """
    Converte os resultados em uma descrição textual.
    """

    ritmo = resultado["ritmo"]
    tonalidade = resultado["tonalidade"]
    energia = resultado["energia"]
    espectro = resultado["espectro"]
    duracao = resultado["duracao"]

    descricao = (
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
        f"{espectro['brilho_estimado']}."
    )

    return descricao


def normalizar_nome_arquivo(nome: str) -> str:
    """
    Cria um nome seguro para pastas e arquivos.
    """

    nome = nome.lower()

    substituicoes = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
    }

    for caractere, substituto in substituicoes.items():
        nome = nome.replace(
            caractere,
            substituto,
        )

    nome = re.sub(
        r"[^a-z0-9]+",
        "_",
        nome,
    )

    nome = nome.strip("_")

    if not nome:
        return "musica"

    return nome


def salvar_resultados(
    resultado: dict,
    descricao: str,
    pasta_output: Path,
    caminho_audio: Path,
) -> Path:
    """
    Cria uma pasta exclusiva para a música e salva os resultados.
    """

    nome_base = normalizar_nome_arquivo(
        caminho_audio.stem
    )

    pasta_musica = pasta_output / nome_base

    contador = 2

    while pasta_musica.exists():
        pasta_musica = (
            pasta_output /
            f"{nome_base}_{contador}"
        )

        contador += 1

    pasta_musica.mkdir(
        parents=True,
        exist_ok=False,
    )

    caminho_json = pasta_musica / "analise.json"
    caminho_txt = pasta_musica / "descricao.txt"

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

    with caminho_txt.open(
        "w",
        encoding="utf-8",
    ) as arquivo_txt:
        arquivo_txt.write(descricao)

    print()
    print(f"Resultados salvos em: {pasta_musica}")
    print(f"JSON: {caminho_json.name}")
    print(f"Descrição: {caminho_txt.name}")

    return pasta_musica


def main() -> None:
    """
    Ponto de entrada do programa.
    """

    pasta_projeto = Path(__file__).resolve().parent
    pasta_input = pasta_projeto / "input"
    pasta_output = pasta_projeto / "output"

    pasta_input.mkdir(exist_ok=True)
    pasta_output.mkdir(exist_ok=True)

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

        resultado = analisar_audio(
            caminho_audio
        )

        descricao = criar_descricao(
            resultado
        )

        pasta_resultado = salvar_resultados(
            resultado=resultado,
            descricao=descricao,
            pasta_output=pasta_output,
            caminho_audio=caminho_audio,
        )

        print()
        print("RESUMO")
        print("------")
        print(descricao)
        print()
        print(
            f"Análise concluída em: "
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
        print("Operação cancelada pelo usuário.")

    except Exception as erro:
        print()
        print("Ocorreu um erro durante a análise:")
        print(f"{type(erro).__name__}: {erro}")


if __name__ == "__main__":
    main()