\# Music Prompt Analyzer



Aplicação em Python para analisar arquivos de música e transformar características do áudio em descrições técnicas e prompts.



\## Estado atual



O programa já consegue:



\- localizar arquivos de áudio na pasta `input`;

\- permitir a escolha da música pelo terminal;

\- analisar MP3, WAV, FLAC, M4A e OGG;

\- calcular duração;

\- estimar BPM;

\- estimar tonalidade e modo;

\- medir energia;

\- analisar brilho e características espectrais;

\- gerar relatórios em JSON e TXT;

\- organizar cada análise em uma pasta própria.

\- analisar a dinâmica da música em segmentos de 10 segundos;

\- identificar trechos suaves e intensos;

\- localizar os principais picos de energia;

\- gerar uma linha do tempo em CSV;

\- gerar um gráfico visual da dinâmica da faixa;

\- comparar segmentos consecutivos da música;

\- detectar crescimentos e quedas de intensidade;

\- identificar possíveis mudanças de seção;

\- medir mudanças de brilho entre os trechos;

\- gerar um relatório de transições em CSV;

\- marcar transições no gráfico de dinâmica;

\- comparar características acústicas entre os segmentos;

\- identificar trechos semelhantes que se repetem;

\- criar mapas estruturais em letras, como A-B-A-C;

\- sugerir possíveis introduções, versos, refrões e pontes;

\- gerar uma matriz de similaridade em CSV;

\- gerar um mapa visual de similaridade;

\- gerar um relatório textual da estrutura aproximada;

\- organização inicial do código em módulos;

\- configurações centralizadas em `src/config.py`;

\- seleção de arquivos isolada em `src/audio_files.py`;

\- funções auxiliares reunidas em `src/helpers.py`;

\- análise básica de áudio isolada em `src/basic_analysis.py`;

\- BPM, tonalidade, energia e espectro separados do arquivo principal.


\## Tecnologias



\- Python

\- Librosa

\- NumPy

\- SoundFile

\- FFmpeg



\## Estrutura

```text

music_prompt_analyzer/
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── audio_files.py
│   ├── basic_analysis.py
│   └── helpers.py
│
├── input/
└── output/





