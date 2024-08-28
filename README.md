# LAMKIT

Code for *SEM paper **Length-Aware Multi-Kernel Transformer for Long Document Classification**

## Required Packages

```bash
torch>=1.9.0
transformers>=4.9.0
scikit-learn>=0.24.1
tqdm>=4.61.1
numpy>=1.20.1
datasets>=1.12.1
nltk>=3.5
scipy>=1.6.3
```

## Project Structure

The project is organized as follows:

- `myexperiments/`: Contains dataset-specific experiment scripts
    - `trainer.py`: The customized Huggingface trainer class
    - `LAMKIT.py`: The proposed LAMKIT model class
    - `{Dataset}.py`: The main program for specific {Dataset}
- `myscripts/`: Contains shell scripts for running experiments with specific hyperparameters
- `README.md`: Project documentation and instructions

## **Run Experiments**

For the easiest way to reproduce the results

1. Modify the dataset path in the myexperiments folder. For example, in `myexperiments/mimic3.py`, locate line 259 and update the `load_dataset()` function call:

```python
train_dataset = load_dataset("path/to/your/dataset")
```

1. Set the hyperparameters in the myscript folder (e.g., `myscripts/mimic-roberta.sh`)
2. Run the shell script file. For example

```shell
sh myscripts/mimic-roberta.sh
```
