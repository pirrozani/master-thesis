# Master Thesis

Fine-Tuning Pretrained Neural Networks for Structured Information Extraction.

This project fine-tunes small language models (Qwen2.5-Instruct) to:

- extract addresses from text
- extract entity names from text
- classify an entity name as **company** or **person**

All examples below use the `qwen-0.5b` model, which is Qwen2.5-0.5B-Instruct

## Requirements

- An NVIDIA GPU with CUDA
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with GPU support (only for the Demonstration)

## 1. Create the Python environment

Run these commands from the project folder.

```bash
uv sync
```

This creates a `.venv` folder with Python 3.13 and installs all packages.


## 2. Get the data

- **Private dataset:** put the CSV file at `data/raw/address.csv`.
- **Public dataset:** download it from Hugging Face:

```bash
uv run hf download ele-sage/person-company-names-classification --type dataset --local-dir data/raw/person-company-names-classification
```

## 3. Preprocess, train, evaluate and infer

Each task has four steps:

1. **Preprocess** turns the raw CSV into train / validation / test splits as Hugging Face datasets.
2. **Train** fine-tunes the model and saves a LoRA adapter.
3. **Evaluate** tests the adapter on the test split.
4. **Infer** opens an interactive prompt: type a text, press Enter, and see the result. Type `exit` to quit.

### Address extraction

```bash
uv run scripts/data/preprocess.py data/raw/address.csv --task address
```

```bash
uv run scripts/address/train.py data/processed/address --model qwen-0.5b --output-dir adapters/address/qwen-0.5b --num-epochs 1
```

```bash
uv run scripts/address/evaluate.py data/processed/address/test --model qwen-0.5b --save-predictions
```

```bash
uv run scripts/address/infer.py -i --model qwen-0.5b
```

### Entity name extraction

```bash
uv run scripts/data/preprocess.py data/raw/address.csv --task entity_name
```

```bash
uv run scripts/entity_name/extraction/train.py data/processed/entity_name/extraction --model qwen-0.5b --num-epochs 1
```

```bash
uv run scripts/entity_name/extraction/evaluate.py data/processed/entity_name/extraction/test --model qwen-0.5b --save-predictions
```

```bash
uv run scripts/entity_name/extraction/infer.py -i --model qwen-0.5b
```

### Company / person classification (private dataset)

```bash
uv run scripts/data/preprocess.py data/raw/address.csv --task entity_type
```

```bash
uv run scripts/entity_name/classification/train.py data/processed/entity_name/classification --model qwen-0.5b --num-epochs 1
```

```bash
uv run scripts/entity_name/classification/evaluate.py data/processed/entity_name/classification/test --model qwen-0.5b --save-predictions
```

```bash
uv run scripts/entity_name/classification/infer.py -i --model qwen-0.5b
```

### Company / person classification (public dataset)

```bash
uv run scripts/data/preprocess.py data/raw/person-company-names-classification/train.csv --task entity_type_external
```

```bash
uv run scripts/entity_name/classification_external/train.py data/processed/entity_name/classification_external --model qwen-0.5b --num-epochs 1
```

```bash
uv run scripts/entity_name/classification_external/evaluate.py data/processed/entity_name/classification_external/test --model qwen-0.5b --save-predictions
```

```bash
uv run scripts/entity_name/classification_external/infer.py -i --model qwen-0.5b
```

### The files are saved in these folders:


Processed data -> `data/processed/<task>/` 

Trained adapter -> `adapters/<task>/qwen-0.5b/` 

Evaluation results -> `outputs/results/<task>/qwen-0.5b.txt`

## 4. Run the full pipeline

The pipeline runs all three tasks on one line of text: it extracts the address and the entity name, then classifies the name as company or person.

```bash
uv run scripts/pipeline/infer.py -i --address-model qwen-0.5b --extraction-model qwen-0.5b --classification-model qwen-0.5b
```

It uses these three adapters, so there is a need to train them first:

- `adapters/address/qwen-0.5b`
- `adapters/entity_name/extraction/qwen-0.5b`
- `adapters/entity_name/classification_external/qwen-0.5b`


For a classifier trained on the private dataset:

```bash
uv run scripts/pipeline/infer.py -i --address-model qwen-0.5b --extraction-model qwen-0.5b --classification-model qwen-0.5b --classification-task classification
```

## 5. Demonstration with Docker

Docker starts two containers:

- **vllm** serves the model and the adapters.
- **app** is the web page and the API.

### Before you start

Docker uses these three trained adapters:

- `adapters/address/qwen-0.5b`
- `adapters/entity_name/extraction/qwen-0.5b`
- `adapters/entity_name/classification_external/qwen-0.5b`

### Build and start

Build and start the containers in detached mode.

```bash
docker compose -f .docker/docker-compose.yml up -d --build
```

### Stop

Stop the containers.

```bash
docker compose -f .docker/docker-compose.yml down
```

### Access the Application
Web page -> `http://localhost:8080`

API docs -> `http://localhost:8080/docs`