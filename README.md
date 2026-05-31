# AP-TP2

## Visão Geral

Este repositório contém o trabalho de grupo da unidade curricular de AP, centrado em classificação de imagens médicas com redes neurais profundas, comparação de arquiteturas e combinação de modelos em ensemble.

O objetivo do projeto foi estudar o impacto de diferentes modelos e estratégias de treino na identificação de quatro classes:

- `Biliary_Leaks`
- `Lithiasis`
- `Normal`
- `Stricture`

O código do projeto também está disponível no GitHub em:

https://github.com/RuiRodrigues17/AP-TP2.git

## Estrutura do Repositório

### Fluxo principal

- `Treino.py` — treino principal do modelo base.
- `Dataloader.py` — carregamento e preparação dos dados do fluxo principal.
- `evaluate.py` — avaliação de checkpoints já treinados.
- `Attention_maps.py` — geração de mapas de atenção para inspeção visual.
- `demo.ipynb` — demonstração interativa com exemplos de execução e visualização.
- `checkpoints/` — modelos treinados e checkpoints guardados durante os experimentos.

### Variações, especialistas e testes

- `TreinoBinarySpecialist.py` — treino de um especialista binário.
- `Dataloader_inicial.py` — versão inicial do carregamento de dados usada em fases anteriores.
- `DataLoaderEnsemble.py` — dataloader adaptado para cenários de ensemble.
- `TreinoEnsemble.py` — treino de variantes usadas em ensemble.
- `Ensemble3.py` — combinação de três modelos.
- `ensembleCascade.py` — abordagem com um especialista.
- `ensembleTest.py` — teste final do ensemble sem treino.
- `ComparacaoDataloader.py` — comparação entre carregamentos/preparações de dados.
- `Confusion_matrix_v6.py` — geração e análise de matrizes de confusão.

## Dados

Os dados não estão na pasta de entrega ZIP. Eles estão disponíveis no repositório do GitHub do link acima.

O layout esperado é o seguinte:

```text
dataset/
  train/
    Biliary_Leaks/
    Lithiasis/
    Normal/
    Stricture/
  val/
    Biliary_Leaks/
    Lithiasis/
    Normal/
    Stricture/
  test/
    Biliary_Leaks/
    Lithiasis/
    Normal/
    Stricture/
```

Os scripts de treino e avaliação assumem esta organização ao chamar `get_dataloaders(...)`.

## Requisitos

Recomendado:

- Python 3.8 ou superior
- Ambiente virtual dedicado
- CUDA/GPU

Instalação típica em Windows:

```bash
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r Requirements.txt
```

## Como Executar

### 1. Treino do modelo principal

```bash
python Treino.py
```

### 2. Treino do especialista

```bash
python TreinoBinarySpecialist.py
```

### 3. Avaliação de um checkpoint

```bash
python evaluate.py --checkpoint checkpoints/best_model_v6.pth --data-path dataset
```

### 4. Teste do ensemble

```bash
python ensembleTest.py
```

### 5. Geração de mapas de atenção

```bash
python Attention_maps.py
```

### 6. Notebook de demonstração

Abrir `demo.ipynb` no Jupyter ou no VS Code e executar as células por ordem.

## Checkpoints

A pasta `checkpoints/` guarda os pesos dos modelos treinados. Entre os ficheiros mais relevantes estão:

- `best_model_v6.pth` (o nosso melhor modelo)
- `convnext-base-flips.pth`
- `best_convnext_v4.pth`
- outros checkpoints usados em testes e comparações

Estes ficheiros são essenciais para reproduzir os resultados sem voltar a treinar do zero.

## Scripts de Comparação e Ensemble

Os scripts abaixo foram usados para comparar abordagens ou testar combinações de modelos:

- `ensembleTest.py` — mistura de probabilidades entre dois modelos já treinados.
- `TreinoEnsemble.py` — treino de estratégias específicas de ensemble.
- `Ensemble3.py` — comparação de três modelos.
- `ensembleCascade.py` — ensemble com um modelo especialista.
- `ComparacaoDataloader.py` — análise de impacto das alterações no dataloader.
- `Confusion_matrix_v6.py` — geração de matriz de confusão para o modelo v6.

## Resultados e Saídas

Dependendo do script executado, os resultados podem ser guardados em:

- `checkpoints/` — pesos dos modelos e melhores épocas
- `results/` — figuras, métricas e saídas auxiliares, quando aplicável
- `wandb/` — logs e métricas experimentais, se o Weights & Biases estiver ativo

## Observações de Uso

- O projeto foi desenvolvido com várias iterações de treino e teste, por isso existem scripts com funções semelhantes mas objetivos diferentes.
- Alguns ficheiros foram mantidos para comparação histórica entre arquiteturas e dataloaders.
- Para executar em GPU, basta garantir que o PyTorch está instalado com suporte CUDA e que o dispositivo é detetado corretamente.

## Resumo Rápido

Se quiser apenas reproduzir a parte principal do projeto, a ordem mais direta é:

1. Preparar os dados na estrutura esperada.
2. Treinar com `Treino.py` ou usar checkpoints já existentes.
3. Avaliar com `evaluate.py`.
4. Consultar `demo.ipynb` e `Attention_maps.py` para visualização.