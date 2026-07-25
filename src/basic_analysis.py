import librosa
import numpy as np

from src.config import (
    NOMES_NOTAS,
    PERFIL_MAIOR,
    PERFIL_MENOR,
)
from src.helpers import converter_para_numero


def classificar_andamento(bpm: float) -> str:
    """
    Classifica o andamento com base no BPM estimado.
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


def classificar_brilho(
    centroide_medio: float,
) -> str:
    """
    Usa o centroide espectral para estimar o brilho.
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
    Calcula a correlação entre dois perfis tonais.
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
    Estima a tonalidade e o modo maior ou menor.
    """

    chroma = librosa.feature.chroma_cqt(
        y=audio_harmonico,
        sr=taxa_amostragem,
    )

    perfil_audio = np.mean(
        chroma,
        axis=1,
    )

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

    nome_nota = NOMES_NOTAS[
        melhor_indice_nota
    ]

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
        "tonalidade_completa": (
            f"{nome_nota} {melhor_modo}"
        ),
        "correlacao": round(
            melhor_correlacao,
            4,
        ),
        "confianca_aproximada": round(
            confianca_aproximada,
            4,
        ),
    }


def analisar_caracteristicas_basicas(
    audio: np.ndarray,
    taxa_amostragem: int,
) -> dict:
    """
    Extrai duração, BPM, tonalidade, energia e espectro.
    """

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

    bpm = converter_para_numero(
        tempo
    )

    print("Estimando tonalidade...")

    tonalidade = estimar_tonalidade(
        audio_harmonico=audio_harmonico,
        taxa_amostragem=taxa_amostragem,
    )

    print("Calculando energia geral...")

    rms = librosa.feature.rms(
        y=audio
    )

    rms_medio = float(
        np.mean(rms)
    )

    rms_maximo = float(
        np.max(rms)
    )

    print(
        "Analisando características "
        "espectrais..."
    )

    centroide = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=taxa_amostragem,
        )
    )

    largura_espectral = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=taxa_amostragem,
        )
    )

    zero_crossing = (
        librosa.feature.zero_crossing_rate(
            audio
        )
    )

    centroide_medio = float(
        np.mean(centroide)
    )

    largura_media = float(
        np.mean(largura_espectral)
    )

    zero_crossing_medio = float(
        np.mean(zero_crossing)
    )

    return {
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
            "bpm_estimado": round(
                bpm,
                2,
            ),
            "classificacao": (
                classificar_andamento(
                    bpm
                )
            ),
            "batidas_detectadas": int(
                len(batidas)
            ),
        },
        "tonalidade": tonalidade,
        "energia": {
            "rms_medio": round(
                rms_medio,
                6,
            ),
            "rms_maximo": round(
                rms_maximo,
                6,
            ),
            "classificacao": (
                classificar_energia(
                    rms_medio
                )
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
            "brilho_estimado": (
                classificar_brilho(
                    centroide_medio
                )
            ),
        },
    }