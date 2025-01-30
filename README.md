# PDF Question Answering System

## Overview

AI-powered system for answering questions about PDF documents using LLMs and vector embeddings.

## Features

- PDF document processing
- Vector embeddings for efficient document retrieval
- Question answering using LLMs
- Web interface using Gradio

## Installation

### Prerequisites

- Python 3.12+
- virtualenv
- pip package manager

### Setup

1. Create virtual environment:

```bash
# Create virtual environment
virtualenv -p python3.12 .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

2. Setup LM Studio server and load model deepseek-r1-distill-qwen-7b

3. Process PDFs and answer questions via CLI or GUI:

```bash
python main.py
```
