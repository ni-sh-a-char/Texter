# Texter  
**A Common Platform for Text analysis in your documents.**  

---  

## Table of Contents  

| Section | Description |
|---------|-------------|
| **[Installation](#installation)** | How to get Texter up and running on your machine. |
| **[Quick‑Start Usage](#quick-start-usage)** | One‑liner commands and basic Python usage. |
| **[API Documentation](#api-documentation)** | Detailed reference for the public classes, functions and CLI. |
| **[Examples](#examples)** | Real‑world snippets that show Texter in action. |
| **[Contributing & Support](#contributing--support)** | How to help improve Texter. |
| **[License](#license)** | Open‑source licensing information. |

---  

## Installation  

Texter is distributed as a pure‑Python package and can be installed via **pip**, **conda**, or directly from source.

### 1. From PyPI (recommended)

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the latest stable release
pip install texter
```

### 2. From Conda‑Forge  

```bash
conda install -c conda-forge texter
```

### 3. From source (development version)

```bash
# Clone the repository
git clone https://github.com/your‑org/texter.git
cd texter

# Install in editable mode with all optional dependencies
pip install -e ".[dev,all]"
```

> **Optional dependencies**  
> - `pdf` – `textract`, `pdfminer.six` – for PDF extraction.  
> - `docx` – `python-docx` – for Microsoft Word files.  
> - `nlp` – `spacy`, `nltk`, `transformers` – for advanced linguistic analysis.  

You can install any subset, e.g.:

```bash
pip install "texter[pdf,docx]"
```

### 4. Verify the installation  

```bash
$ texter --version
Texter 2.3.1
```

---  

## Quick‑Start Usage  

Texter can be used **as a command‑line tool**, **as a library**, or **as a web‑service** (via the optional `flask` extra). Below are the most common entry points.

### 2.1 CLI  

```bash
# Basic analysis of a single file
texter analyze path/to/file.txt

# Batch processing of a directory (recursive)
texter analyze ./documents --recursive --output results.json

# Export a summary report (CSV, JSON, or Markdown)
texter report results.json --format markdown > report.md
```

#### CLI Options Overview  

| Flag | Description |
|------|-------------|
| `-h, --help` | Show help message and exit. |
| `-r, --recursive` | Walk sub‑directories when a folder is supplied. |
| `-o, --output <file>` | Write the raw analysis JSON to *file*. |
| `--lang <code>` | Force language detection (e.g., `en`, `de`, `fr`). |
| `--pipeline <name>` | Choose a pre‑configured analysis pipeline (`basic`, `nlp`, `semantic`). |
| `--threads <n>` | Number of worker threads for parallel processing (default: CPU count). |

### 2.2 Python Library  

```python
from texter import Texter, pipelines

# Load a document (any supported format)
doc = Texter.load("reports/annual_report.pdf")

# Run a built‑in pipeline (basic stats + language detection)
result = doc.analyze(pipelines.basic)

# Access results
print("Word count:", result.stats.word_count)
print("Detected language:", result.language.iso_code)

# Save the JSON output
result.to_json("annual_report_analysis.json")
```

#### Common workflow  

```python
from texter import Texter, Analyzer, pipelines

# 1️⃣ Load one or many documents
paths = ["doc1.txt", "doc2.docx", "presentation.pdf"]
documents = [Texter.load(p) for p in paths]

# 2️⃣ Choose a pipeline (or build a custom one)
pipeline = pipelines.nlp   # includes tokenisation, POS, NER, sentiment

# 3️⃣ Analyse in parallel (optional)
analyzer = Analyzer(pipeline=pipeline, workers=4)
results = analyzer.run(documents)

# 4️⃣ Post‑process / visualise
for r in results:
    print(r.summary())
```

### 2.3 Web Service (optional)  

```bash
# Install the optional Flask extra
pip install "texter[web]"

# Start the server
texter serve --host 0.0.0.0 --port 8080
```

The service exposes a simple REST API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload a file (multipart) and receive JSON analysis. |
| `GET`  | `/api/pipelines` | List available pipelines. |
| `GET`  | `/api/health` | Health‑check endpoint. |

---  

## API Documentation  

> **Version:** 2.3.1 (latest)  
> **Generated with:** `sphinx` + `autodoc` (see `docs/` folder for full HTML docs)

### 3.1 Core Classes  

| Class | Purpose | Important Methods |
|-------|---------|-------------------|
| `Texter` | High‑level wrapper for a single document. Handles loading, format detection, and lazy parsing. | `load(path)`, `text`, `metadata`, `analyze(pipeline)`, `to_json(path)` |
| `Analyzer` | Orchestrates batch processing, thread‑pool management, and pipeline execution. | `run(documents)`, `add_pipeline(name, pipeline)`, `close()` |
| `Pipeline` | Immutable collection of `Processor` objects executed sequentially. | `add(processor)`, `execute(document)`, `name` |
| `Result` | Container for the output of a single analysis run. Provides helpers for serialization and summarisation. | `to_dict()`, `to_json(path)`, `summary()`, `stats`, `language`, `entities` |

### 3.2 Processors (building blocks)  

All processors inherit from `texter.processors.base.Processor`. They receive a `Document` object and may augment it with new attributes.

| Processor | Category | Typical Use |
|-----------|----------|-------------|
| `Tokeniser` | NLP | Splits raw text into tokens. |
| `POSTagger` | NLP | Part‑of‑speech tagging (spaCy backend). |
| `NER` | NLP | Named‑entity recognition. |
| `SentimentAnalyzer` | NLP | Polarity & subjectivity scores (VADER / TextBlob). |
| `LanguageDetector` | Meta | Detects language using `langdetect` or `fasttext`. |
| `Readability` | Stats | Computes Flesch‑Kincaid, Gunning Fog, etc. |
| `KeywordExtractor` | NLP | RAKE / YAKE based keyword extraction. |
| `TopicModeler` | Advanced | LDA / BERTopic (requires `transformers`). |
| `CustomProcessor` | Extensible | Subclass to implement your own logic. |

#### Example: Creating a custom processor  

```python
from texter.processors.base import Processor

class WordLengthStats(Processor):
    """Adds average, min and max word length to the result."""
    def process(self, doc):
        lengths = [len(tok) for tok in doc.tokens]
        doc.stats.word_length = {
            "avg": sum(lengths) / len(lengths),
            "min": min(lengths),
            "max": max(lengths),
        }
        return doc
```

Add it to a pipeline:

```python
from texter import Pipeline
pipeline = Pipeline(name="custom")
pipeline.add(WordLengthStats())
pipeline.add(pipelines.basic)   # chain with existing processors
```

### 3.3 Helper Functions  

| Function | Module | Description |
|----------|--------|-------------|
| `load(path)` | `texter.io` | Detects file type and returns a `Texter` instance. |
| `detect_language(text)` | `texter.utils.lang` | Returns a `Language` object (`iso_code`, `name`, `confidence`). |
| `export_to_csv(results, path)` | `texter.io.export` | Serialises a list of `Result` objects to CSV. |
| `visualise_entities(result, output="html")` | `texter.visualisation` | Generates an interactive HTML view of NER tags. |

### 3.4 Configuration  

All defaults are stored in `texter.config.DEFAULTS`. You can override them globally:

```