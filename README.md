# AhmedBsher-PRODIGY_TrackCode_Task-1
Fine-tune OpenAI's GPT-2 model on a custom dataset for text generation tasks. This repository includes scripts for tokenizing the dataset, fine-tuning the model, and generating text based on given prompts. Ideal for creating AI-driven text responses tailored to specific contexts.


# Fine-Tuning GPT-2 for Custom Text Generation

This repository contains code to fine-tune OpenAI's GPT-2 model on a custom dataset and generate text based on given prompts. The project is set up to handle a small dataset of queries and responses, demonstrating how to tokenize the data, fine-tune the model, and generate text.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
  - [tokenize_and_finetune.py](#tokenize_and_finetunepy)
  - [generate_text.py](#generate_textpy)
- [Dataset](#dataset)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/fine-tune-gpt2.git
    cd fine-tune-gpt2
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```


## Usage

1. Prepare your dataset as described in the [Dataset](#dataset) section.

2. Run the `tokenize_and_finetune.py` script to fine-tune the GPT-2 model on your dataset:
    ```sh
    python tokenize_and_finetune.py
    ```

3. After fine-tuning, use the `generate_text.py` script to generate text based on a prompt:
    ```sh
    python generate_text.py
    ```

    
## Scripts
- [Tokenize and Fine-tune Script](tokenize_and_finetune.py)
- [Generate Text script](generate_text.py)


## Dataset
-- the .csv files


## Project Structure
```
Text Generation with GPT-2/
 ├── dataset/
│ ├── train_dataset.csv
│ ├── val_dataset.csv
│ └── test_dataset.csv
├── logs/
├── results/
├── env/
├── tokenize_and_finetune.py
├── generate_text.py
├── requirements.txt
└── README.md
```

## Acknowledgements

- Hugging Face for the Transformers library.
- OpenAI for developing GPT-2.

##License
- This project is licensed under the MIT License.

