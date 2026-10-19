# Text Summarizer

A simple text summarization web app built with FastAPI and a local T5 model. Enter text in the browser and the app returns a generated summary.

## Features

- Browser-based text input and summary display
- FastAPI endpoint at `POST /summarize`
- Runs inference on CUDA when available, otherwise on CPU

## Project structure

```text
.
├── app.py
├── index.html
└── saved_summary_model/
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

The model files in `saved_summary_model/` are required because the application loads the model from that local directory.

## Requirements

- Python 3.10 or newer
- PyTorch
- FastAPI and Uvicorn
- Transformers and SentencePiece

Install the required packages:

```bash
python -m pip install fastapi uvicorn jinja2 torch transformers sentencepiece
```

For GPU inference, install the PyTorch build that matches your CUDA setup. The command above installs the default PyTorch build for your environment.

## Run locally

Clone or download this repository, then run these commands from the project directory:

```bash
python -m pip install fastapi uvicorn jinja2 torch transformers sentencepiece
python -m uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The summarization API accepts JSON in this format:

```json
{
  "dialogue": "Text to summarize goes here."
}
```

It returns:

```json
{
  "summary": "Generated summary."
}
```

## Model hosting note

`saved_summary_model/model.safetensors` is approximately 242 MB, which exceeds GitHub's 100 MB limit for regular Git files. To include it in this repository, configure [Git LFS](https://git-lfs.com/) before committing the model. Alternatively, host the model files separately and update `model_path` in `app.py` and these setup instructions accordingly.

Check the model's license and redistribution terms before publishing its files.
