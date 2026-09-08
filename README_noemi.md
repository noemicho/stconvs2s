# Reprodução do Treinamento STConvS2S - Radar Sumaré + Alerta Rio

Este documento descreve os passos necessários para reproduzir o treinamento da arquitetura STConvS2S-C utilizando imagens do Radar Meteorológico do Sumaré e precipitação das estações do Alerta Rio.

A versão atual reúne:

- funções de perda mascaradas e ponderadas;
- balanced sampler por intensidade de precipitação;
- métricas por faixa de intensidade e horizonte;
- suporte aos targets das estações do Alerta Rio;
- possibilidade de alternar entre Alerta Rio e WebSirene.

## 1. Dataset utilizado

O treinamento utiliza o dataset anual em formato `memmap`, com imagens de radar e alvos/máscaras por ano.

- Não é necessário clonar o repositório do atmoseer, já que iremos usar apenas os arquivos dos zips quando rodar o modelo.

Diretório do dataset:

`atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano`

- Adicionar os zips que enviei pelo Teams, nessa pasta, como no exemplo a seguir:

Estrutura esperada:

```bash
radar_sumare_2012_2024_15min_256_por_ano/
├── year=2012/
│   ├── radar_frames.dat
│   ├── radar_timestamps.npy
│   ├── metadata.json
│   │
│   ├── Y_alertario.dat
│   ├── M_alertario.npy
│   ├── targets_alertario_metadata.json
│   │
│   ├── Y_all.dat
│   ├── M_all.dat
│   └── targets_metadata.json
├── year=2013/
│   └── ...
...
└── year=2024/
    └── ...
```

- Ao executar o 'main.py', pelo comando do Passo '3. Rodar o treinamento', o código carrega os arquivos `memmap` de cada ano, monta os conjuntos de treino, validação e teste e inicia o treinamento.

O arquivo `radar_frames.dat` contém as imagens de radar agregadas em intervalos de 15 minutos e redimensionadas para `256 x 256` pixels.
O arquivo `Y_alertario.dat` contém os alvos esparsos de precipitação das estações Alerta Rio, armazenados na escala `log1p(mm/15min)`.
O arquivo `M_alertario.dat` contém a máscara binária indicando onde existe observação pluviométrica disponível.

Os arquivos `Y_all.dat`, `M_all.dat` e `targets_metadata.json` correspondem à WebSirene e foram mantidos para permitir comparações futuras.

## 2. Clonar e instalar o repositório STConvs2s

Clonar a branch `semana-4-alertario-integracao` do repositório:

```bash
git clone -b semana-4-alertario-integracao https://github.com/noemicho/stconvs2s.git
cd stconvs2s
```

A branch `semana-4-alertario-integracao` foi criada a partir da branch `semana-3-sampler-metricas` e recebeu, por merge, as alterações da branch `semana-2-losses-mascaradas-ponderadas`. Dessa forma, a branch atual preserva as funcionalidades das Semanas 2 e 3 e adiciona o suporte aos targets do Alerta Rio.

As dependências do projeto estão especificadas em `config/environment.yml`. 

```bash
conda env create -f config/environment.yml
conda activate pytorch
```

Alternativa com pip:
```bash
python -m pip install torch torchvision matplotlib ipykernel h5py pandas xarray dask bottleneck statsmodels scikit-learn cartopy
```

## 3. Rodar o treinamento

O treinamento é executado pelo arquivo `main.py`, utilizando o modelo `stconvs2s-c`.

Comando utilizado:

```bash
nohup python -u main.py \
  -m stconvs2s-c \
  -dsp /atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --years 2012-2024 \
  --target-source alertario \
  --balanced-sampler \
  --sampler-thresholds 1.25,6.25,12.5 \
  --output-channels 1 \
  --loss weighted-mae \
  --loss-weights 1,5,10,20 \
  -b 8 \
  -e 10 \
  -i 1 \
  -w 0 \
  -s 5 \
  -c 0 \
  --verbose \
  > resultado_stconvs2s_2012_2024_alertario.log 2>&1 &
```

## 4. Parâmetros principais

| Parâmetro              |    Valor usado   | Descrição                          |
| -----------            | -------------:   | ---------------------------------- |
| `-m`                   |  `stconvs2s-c`   | Modelo utilizado                   |
| `-dsp`                 | dataset memmap   | Caminho do dataset                 |
| `--years`              |    `2012-2024`   | Anos usados no treinamento         |
| `--target-source`      |      `alertario` | Fonte das estações                 |
| `--balanced-sampler`   |          ativado | Ativa o balanceamento das amostras |
| `--sampler-thresholds` | `1.25,6.25,12.5` | Limites em mm/15min                |
| `--output-channels`    |              `1` | Número de canais previstos         |
| `--loss`               |   `weighted-mae` | Loss mascarada e ponderada         |
| `--loss-weights`       |      `1,5,10,20` | Pesos das quatro faixas            |
| `-b`                   |            `4`   | Tamanho do batch                   |
| `-e`                   |            `2`   | Número de épocas                   |
| `-i`                   |            `1`   | Número de iterações                |
| `-w`                   |            `0`   | Número de workers do DataLoader    |
| `-s`                   |            `5`   | Stride das janelas temporais       |
| `-c`                   |            `0`   | GPU utilizada                      |
| `--verbose`            |        ativado   | Exibe progresso no log             |


## 5. Acompanhar o treinamento

Para acompanhar a execução:

```bash
tail -f resultado_stconvs2s_2012_2024_alertario.log
```

Para verificar se o processo ainda está rodando:

```bash
pgrep -af "main.py"
```

## 6. Saídas geradas

Os checkpoints são salvos automaticamente no diretório de saída do projeto, em uma estrutura semelhante a:

```bash
/stconvs2s/output/full-dataset/checkpoints/stconvs2s-c/
```

O treinamento também gera arquivos de log com as métricas globais, métricas por horizonte e métricas por faixa de intensidade de precipitação, calculadas após o retorno dos valores para mm/15min.


## 7. Montagem dos targets e máscaras do Alerta Rio

Os arquivos de targets e máscaras do Alerta Rio são gerados pelo script:
 
```bash
scripts/radar_sumare/gerar_targets_masks_alertario_memmap_por_ano.py
```

Exemplo de execução:

```bash
python scripts/radar_sumare/gerar_targets_masks_alertario_memmap_por_ano.py \
  --alertario-root /home/noemi/atmoseer/data/alertario/pluviometricos_parquet \
  --mapping /home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano/mapeamento_pixel_estacao_alertario.csv \
  --radar-root /home/noemi/atmoseer/data/datasets/radar_sumare_2012_2024_15min_256_por_ano \
  --year-start 2012 \
  --year-end 2024 \
  --height 256 \
  --width 256
```

Para cada ano, o script gera:

```bash
year=AAAA/
├── Y_alertario.dat
├── M_alertario.dat
└── targets_alertario_metadata.json
```

Parâmetros do script:

| Parâmetro          |        Valor utilizado | Descrição                                                       |
| ------------------ | ---------------------: | --------------------------------------------------------------- |
| `--alertario-root` | diretório dos Parquets | Diretório com os dados pluviométricos do Alerta Rio             |
| `--mapping`        |            arquivo CSV | Mapeamento entre as estações do Alerta Rio e os pixels do radar |
| `--radar-root`     |   diretório do dataset | Diretório com os dados anuais do radar                          |
| `--year-start`     |                 `2012` | Primeiro ano processado                                         |
| `--year-end`       |                 `2024` | Último ano processado                                           |
| `--height`         |                  `256` | Altura da grade espacial de saída                               |
| `--width`          |                  `256` | Largura da grade espacial de saída                              |



## 8. Observação sobre unidades

A unidade nativa da precipitação é:

```bash
mm/15min
```

As faixas de intensidade também são apresentadas em taxa horária equivalente:

```bash
mm/h equivalente = 4 × mm/15min
```

Assim:

| Faixa          |     Unidade nativa | Unidade equivalente |
| -------------- | -----------------: | ------------------: |
| Chuva fraca    |    < 1,25 mm/15min |            < 5 mm/h |
| Chuva moderada | 1,25–6,25 mm/15min |           5–25 mm/h |
| Chuva forte    | 6,25–12,5 mm/15min |          25–50 mm/h |
| Chuva extrema  |    > 12,5 mm/15min |           > 50 mm/h |


## 9. Exemplo de log de saída do treinamento
