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

├── README.md

├── input/

└── output/

