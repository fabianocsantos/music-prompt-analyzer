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

\- gerar um gráfico visual da dinâmica da faixa.


\## Tecnologias



\- Python

\- Librosa

\- NumPy

\- SoundFile

\- FFmpeg



\## Estrutura



```text

music\_prompt\_analyzer/

├── app.py

├── requirements.txt

├── analise.json

├── descricao.txt

├── dinamica.csv

├── grafico_dinamica.png

├── README.md

├── input/

└── output/


