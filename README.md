# AP-TP2

Resumo
-
Projeto do Módulo 2 (AP) — Trabalho de Grupo: implementação de modelos e ensemble para classificação. Este repositório contém código, dados (quando possíveis), scripts de treino e avaliação, e exemplos para reproduzir os resultados obtidos.

Conteúdo do repositório
-
- `Treino.py` — script principal de treino (ver parâmetros no cabeçalho).
- `TreinoBinarySpecialist.py` — variante de treino para especialistas binários.
- `TreinoEnsemble.py`, `Ensemble3.py`, `ensembleCascade.py`, `ensembleTest.py` — implementações e testes de ensemble.
- `Dataloader.py`, `Dataloader_inicial.py`, `DataLoaderEnsemble.py`, `ComparacaoDataloader.py` — carregamento e preparação de dados.
- `evaluate.py` — script para avaliar modelos a partir de checkpoints.
- `Confusion_matrix_v6.py`, `confusion_matrix_v6.py` — utilitários para gerar matrizes de confusão e métricas.
- `Attention_maps.py` — geração de mapas de atenção (se aplicável).
- `demo.ipynb` — notebook com exemplos de uso e visualizações.
- `checkpoints/` — pasta com checkpoints (não commitar grandes modelos; incluir apenas o necessário para reprodução).

Requisitos e ambiente
-
Recomendado: Python 3.8+ e ambiente virtual. Instalar dependências com:

Windows / PowerShell

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Arquivo de dependências
-
Veja `requirements.txt` para a lista completa de pacotes e versões usadas.

Executar os experimentos (exemplos)
-
1) Treino de um modelo (exemplo genérico)

```
python Treino.py --epochs 50 --batch-size 32 --data-path PATH_PARA_DADOS --output checkpoints/saida.pth
```

2) Treino de ensemble (exemplo)

```
python TreinoEnsemble.py --models model1,model2 --data-path PATH_PARA_DADOS --output checkpoints/ensemble.pth
```

3) Avaliação a partir de checkpoint

```
python evaluate.py --checkpoint checkpoints/saida.pth --data-path PATH_PARA_DADOS --out-dir results/
```

4) Executar o notebook de demonstração

Abra `demo.ipynb` no Jupyter / VSCode e execute as células na ordem.

Parâmetros e seeds
-
Para reprodução exacta, fixe as seeds (quando suportado pelos scripts) e especifique `--seed 42` (ou outra seed usada nos relatórios). Se os scripts não expõem seed via CLI, edite o início do script para definir a seed global (NumPy, random e framework ML usado).

Dados
-
Se os dados não estiverem incluídos neste repositório, siga as instruções no relatório (ou no cabeçalho dos scripts) para obter e preparar os datasets. Os argumentos `--data-path` usados nos exemplos devem apontar para a pasta com as imagens/CSV organizados conforme esperado pelo `Dataloader.py`.

Uso em Google Colab / estrutura alternativa
-
Muitos dos treinos e experimentos foram executados em Google Colab, pelo que existe uma estrutura alternativa de ficheiros/paths usada nos notebooks. Notas importantes para reproduzir em Colab:

- Ativar GPU: Runtime → Change runtime type → Hardware accelerator → GPU.
- Montar o Google Drive para ler/gravar dados e checkpoints:

```
from google.colab import drive
drive.mount('/content/drive')
```

- Clonar o repositório no Colab ou enviar o zip para o Drive e descompactar:

```
!git clone <URL_DO_REPO>
%cd AP-TP2
pip install -r requirements.txt
```

- Paths típicos usados no Colab:
	- Dados: `/content/drive/MyDrive/path_para_dados/`
	- Checkpoints: `/content/drive/MyDrive/AP-TP2/checkpoints/`
	- Resultados/figuras: `/content/drive/MyDrive/AP-TP2/results/`

- Para garantir que os scripts escrevem direta­mente no Drive (evita perda de dados ao terminar a sessão), passe `--output /content/drive/MyDrive/AP-TP2/checkpoints/saida.pth` ou configure `--out-dir` para uma pasta no Drive.

- O `demo.ipynb` pode ter células que assumem paths do Drive; verifique e ajuste os caminhos locais antes de executar.

- Aviso: alguns checkpoints e ficheiros de resultados referenciados no relatório podem não estar incluídos no repositório porque ficaram apenas no Drive do autor. Nesse caso, os ficheiros essenciais para reprodução devem ser adicionados ao zip de submissão (ver secção "Como gerar o ficheiro ZIP para submissão").

Ficheiros de saída e onde encontrar resultados
-
- Checkpoints: `checkpoints/`
- Matrizes de confusão e figuras: `results/` (ou pasta indicada com `--out-dir`)
- Logs de treino: caso os scripts escrevam logs, estarão na pasta `logs/` ou conforme o parâmetro de saída.

