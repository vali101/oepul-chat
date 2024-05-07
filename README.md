# Oepul Chat - A Chatbot interface for 

## Installation

To run the project, follow these steps:

1. **Install Poetry**: If you don't have Poetry installed, you can do so by following the official instructions [here](https://python-poetry.org/docs/).

2. **Install Dependencies**: Use Poetry to install the project dependencies.

    ```bash
    poetry install
    ```

3. Create a `.env` file in the project directory and add your OpenAI API key like in .env.default


## Usage

To use the system run `poetry run python main.py` with one of the following arguments:

### Download Data
- `--download`: Download necessary data for the chatbot.

### Building Indices
- `--build-summary-index`: Create a summary index.
- `--build-normal-index`: Create a normal index with only OEPUL documents. 
- `--build-oepul-base-reader-index`: Create only the OEPUL base reader index.
- `--build-indices`: Create all indices.

### Evaluation
- `--evaluate-retriever`: Evaluate the chatbot's performance.
  - `--combinations`: Evaluate the chatbot with different combinations.
  - `--file-path [path]`: Specify the file path for evaluation.
  - `--k [number]`: Specify the number of documents retrieved.

### Querying
- `--query [text]`: Query the chatbot with the provided text.
  - `--k [number]`: Specify the number of documents retrieved.
- `--summary-query [text]`: Query summaries with the provided text.

### Chatting
- `--chat`: Engage in a chat session with the chatbot.

### Debugging
- `--debug`: Enable debug logging.

## Example Usages
- To download data: `python program.py --download`
- To build indices: `poetry run python app.py --build-indices`
- To engage in a chat session: `poetry run python app.py --chat`
- To query the chatbot: `poetry run python app.py --query "Unter welchen Umständen ist Anbindehaltung möglich?"`
- To evaluate the chatbot: `poetry run python app.py --evaluate`

## Repo Structure

This repository primarily consists of the oepul_chat folder, housing two submodules:

* index_build: Contains code for constructing various indices.
* readers: Encompasses different readers for handling data.

Additionally, there are smaller modules dedicated to evaluating the RAG system and utilities for file loading.

Upon executing `--download`, a new folder named data will be generated. Subsequently, when building an index, it will be stored within the indices folder.
