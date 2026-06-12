# MtlSAR

This repository provides the PyTorch implementation of **MtlSAR**, the model used in our paper:

> MtlSAR: A Typed Event Representation Framework for Unified Search and Recommendation

The code includes the model definition, data loaders, training loop, and evaluation code for the search and recommendation tasks.

## Requirements

The experiments were run with Python 3 and PyTorch. The main dependencies are:

- PyTorch
- NumPy
- pandas
- PyYAML
- tqdm

Install the packages with the CUDA/PyTorch version that matches your machine. For example:

```bash
pip install numpy pandas pyyaml tqdm
```

Please install PyTorch from the official instructions for your CUDA version.

## Repository Structure

```text
.
|-- main.py
|-- README.md
`-- MtlSAR
    |-- models
    |   |-- const.py
    |   |-- inputs.py
    |   |-- layers.py
    |   `-- MtlSAR.py
    `-- utils
        |-- dataset.py
        |-- runner.py
        |-- sampler.py
        `-- util.py
```

## Data

The code expects preprocessed pickle files. By default, dataset paths are defined in `MtlSAR/models/const.py`:

```text
../JDsearch/processed_data
../KuaiSAR/processed_data
```

Each processed data directory should contain:

```text
user_vocab.pkl
query_vocab.pkl
src_train.pkl
src_val.pkl
src_test.pkl
rec_train.pkl
rec_val.pkl
rec_test.pkl
user_feats_vocab.pkl
item_feats_vocab.pkl
search_vocab.pkl
```

The training files are loaded by pandas from pickle format. The current sampler uses the following fields: `user_id`, `item_id`, `neg_items`, `rec_his_num`, `src_session_his_num`, and task-specific fields such as `keyword`, `weak_neg_items`, and `behavior_type`.

## Training

Run training from the repository root:

```bash
python main.py --data JDsearch --device cuda:0
```

For KuaiSAR:

```bash
python main.py --data KuaiSAR --device cuda:0
```

Common options:

```bash
python main.py \
	--data JDsearch \
	--device cuda:0 \
	--random_seed 20251211 \
	--epoch 100 \
	--batch_size 1024 \
	--eval_batch_size 512 \
	--lr 0.01
```

Checkpoints and logs are written to:

```text
output/<dataset>/MtlSAR/checkpoints/<run_name>/best.pt
output/<dataset>/MtlSAR/logs/<run_name>.log
```

If `--time` is not specified, a timestamp is used as the run name.

## Evaluation

To evaluate a saved checkpoint:

```bash
python main.py \
	--data JDsearch \
	--device cuda:0 \
	--train 0 \
	--test_path output/JDsearch/MtlSAR/checkpoints/<run_name>/best.pt
```

The runner reports HR and NDCG at `1`, `5`, `10`, `30`, and `50` for both recommendation and search. Model selection uses the average of recommendation and search `NDCG@5` on the validation set.

## Reproducibility Notes

The entry script sets random seeds for Python, NumPy, and PyTorch. Results may still vary slightly across hardware, CUDA versions, and PyTorch versions.

For debugging autograd issues, add:

```bash
--detect_anomaly
```

This option slows down training and is not needed for normal runs.

## Citation

If you use this code, please cite the paper. The BibTeX entry will be added after publication.
