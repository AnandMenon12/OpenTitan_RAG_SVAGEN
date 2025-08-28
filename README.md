# OpenTitan RAG SVA Generator

A Retrieval-Augmented Generation (RAG) system for automatically generating SystemVerilog Assertions (SVA) for OpenTitan IP blocks.

## Overview

This system combines web scraping, semantic search, and large language models to generate high-quality SystemVerilog assertions for OpenTitan hardware IP blocks. It downloads documentation from the OpenTitan website, processes it into a searchable knowledge base, and uses AI to generate contextually relevant SVA properties.

## Features

- **Web-based Documentation Ingestion**: Automatically downloads and processes OpenTitan IP documentation from the official website
- **Semantic Search**: Uses FAISS vector database for efficient similarity search across documentation
- **AI-Powered Generation**: Leverages Qwen2-7B-Instruct model for generating SVA properties
- **Caching System**: Stores processed documentation and embeddings for faster subsequent runs
- **Logging**: Automatically logs all generation sessions to JSON files for analysis
- **Single-Run Mode**: Executes once per invocation, perfect for automation workflows

## Supported IP Blocks

- UART
- I2C
- KMAC
- LC_CTRL
- OTBN
- SYSRST_CTRL

## Requirements

- Python 3.9+
- See `requirements.txt` for complete dependency list

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AnandMenon12/OpenTitan_RAG_SVAGEN.git
cd OpenTitan_RAG_SVAGEN
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the SVA generator:
```bash
python opentitan_sva_generator.py
```

The system will:
1. Display available IP blocks
2. Prompt for an IP name (e.g., "i2c", "uart")
3. Ask for a query describing the desired assertions
4. Generate and display SVA properties
5. Save the session to a JSON log file

### Example

```
Enter IP name: i2c
Query for i2c: Generate assertions for start and stop conditions
```

Output will include SystemVerilog assertions like:
```systemverilog
property start_condition_before_transmit;
  @(posedge clk_i) disable iff (!rst_ni)
  FDATA.START |=> TX_BYTE : ~TX_BYTE;
endproperty
assert_start_before_transmit: assert property (start_condition_before_transmit);
```

## Architecture

### Components

1. **OpenTitanIngester**: Downloads and processes OpenTitan documentation from web sources
2. **EmbeddingManager**: Creates and manages FAISS vector indices for semantic search
3. **SVAGenerator**: Uses Qwen2-7B-Instruct model to generate SVA properties
4. **OpenTitanSVASystem**: Orchestrates the entire pipeline

### Data Flow

1. Web documentation is scraped and chunked
2. Text chunks are embedded using BGE-base-en-v1.5
3. FAISS index enables fast similarity search
4. Query-relevant context is retrieved
5. LLM generates SVA properties based on context and query
6. Results are displayed and logged

## Caching

The system caches:
- Downloaded documentation (`cache/docs/`)
- FAISS indices (`cache/faiss/`)
- Session logs (`cache/logs/`)

## Configuration

Key parameters can be modified in the code:
- `model_name`: Language model for generation (default: "Qwen/Qwen2-7B-Instruct")
- `embedding_model`: Embedding model (default: "BAAI/bge-base-en-v1.5")
- `cache_dir`: Directory for cached data (default: "./cache")

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenTitan project for hardware IP documentation
- Hugging Face for model hosting
- FAISS for efficient similarity search
- Qwen team for the language model
