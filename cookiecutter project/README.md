# mloproject

A short description of the project.

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mloproject  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

# Machine Learning Operations Project

This project focuses on developing and operationalizing machine learning models for a specific task using corrupted MNIST data.

## Getting Started

These instructions will help you set up and run the project.

### Prerequisites

- Python (version 3.10.x)
- Anaconda or Miniconda for managing virtual environments

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    conda create --name mlo python=3.10
    conda activate mlo
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Scripts

1. **Data Preprocessing**

    ```bash
    python mloproject/mloproject/data/make_dataset.py
    ```

2. **Training the Model**

    ```bash
    python mloproject/mloproject/models/train_model.py --lr 0.001
    ```

3. **Evaluating the Model**

    ```bash
    python mloproject/mloproject/models/evaluate_model.py trained_model.pt
    ```

4. **Making Predictions**

    ```bash
    python mloproject/mloproject/models/predict_model.py pretrained_model.pt --data_folder data/raw
    ```