from pathlib import Path

import librosa

from src.basic_analysis import (
    analisar_caracteristicas_basicas,
)
from src.dynamics import (
    analisar_dinamica,
    detectar_transicoes,
)
from src.structure_analysis import (
    analisar_repeticoes,
)


def analisar_audio(
    caminho_audio: Path,
) -> dict:
    """
    Carrega o áudio e coordena todas as etapas de análise.
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