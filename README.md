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


KuaiSAR Rec (999 negatives):
| Metric  | DIN             | SASRec          | CL4SRec                | NMTR            | NextIP          | EIDP            | JSR             | SESRec          | USER                   | UnifiedSSR      | UniSAR                 | MtlSAR               |
| ------- | --------------- | --------------- | ---------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---------------------- | --------------- | ---------------------- | -------------------- |
| HR@1    | 0.0109 ± 0.0009 | 0.0104 ± 0.0007 | 0.0117 ± 0.0008        | 0.0101 ± 0.0011 | 0.0095 ± 0.0008 | 0.0099 ± 0.0010 | 0.0106 ± 0.0009 | 0.0105 ± 0.0007 | <u>0.0122</u> ± 0.0007 | 0.0103 ± 0.0006 | 0.0120 ± 0.0006        | **0.0137**\* ± 0.0007 |
| HR@5    | 0.0408 ± 0.0014 | 0.0406 ± 0.0008 | <u>0.0452</u> ± 0.0015 | 0.0394 ± 0.0014 | 0.0385 ± 0.0009 | 0.0396 ± 0.0025 | 0.0413 ± 0.0015 | 0.0399 ± 0.0014 | 0.0434 ± 0.0016        | 0.0404 ± 0.0010 | 0.0447 ± 0.0012        | **0.0490**\* ± 0.0022 |
| HR@10   | 0.0668 ± 0.0011 | 0.0684 ± 0.0015 | 0.0729 ± 0.0022        | 0.0649 ± 0.0015 | 0.0653 ± 0.0015 | 0.0660 ± 0.0018 | 0.0682 ± 0.0011 | 0.0674 ± 0.0015 | 0.0698 ± 0.0026        | 0.0676 ± 0.0014 | <u>0.0735</u> ± 0.0011 | **0.0781**\* ± 0.0013 |
| NDCG@5  | 0.0252 ± 0.0011 | 0.0254 ± 0.0009 | <u>0.0284</u> ± 0.0017 | 0.0244 ± 0.0010 | 0.0238 ± 0.0012 | 0.0241 ± 0.0009 | 0.0258 ± 0.0010 | 0.0247 ± 0.0016 | 0.0273 ± 0.0018        | 0.0249 ± 0.0019 | 0.0281 ± 0.0011        | **0.0314**\* ± 0.0014 |
| NDCG@10 | 0.0330 ± 0.0011 | 0.0346 ± 0.0010 | 0.0372 ± 0.0026        | 0.0329 ± 0.0012 | 0.0323 ± 0.0013 | 0.0327 ± 0.0012 | 0.0349 ± 0.0019 | 0.0332 ± 0.0012 | 0.0354 ± 0.0009        | 0.0343 ± 0.0009 | <u>0.0376</u> ± 0.0012 | **0.0408**\* ± 0.0012 |

JDsearch Rec (999 negatives):
| Metric  | DIN             | SASRec          | CL4SRec         | NMTR            | NextIP          | EIDP            | JSR             | SESRec                 | USER            | UnifiedSSR      | UniSAR                 | MtlSAR               |
| ------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---------------------- | --------------- | --------------- | ---------------------- | -------------------- |
| HR@1    | 0.5908 ± 0.0038 | 0.4907 ± 0.0076 | 0.5974 ± 0.0098 | 0.5721 ± 0.0105 | 0.5679 ± 0.0095 | 0.4528 ± 0.0068 | 0.5406 ± 0.0119 | <u>0.6082</u> ± 0.0046 | 0.5819 ± 0.0057 | 0.5386 ± 0.0063 | 0.6071 ± 0.0039        | **0.6491**\* ± 0.0040 |
| HR@5    | 0.7269 ± 0.0038 | 0.7075 ± 0.0045 | 0.7131 ± 0.0041 | 0.7104 ± 0.0048 | 0.7043 ± 0.0051 | 0.7030 ± 0.0041 | 0.6978 ± 0.0035 | 0.7487 ± 0.0039        | 0.7331 ± 0.0036 | 0.6868 ± 0.0047 | <u>0.7495</u> ± 0.0040 | **0.7521** ± 0.0031  |
| HR@10   | 0.7417 ± 0.0039 | 0.7260 ± 0.0034 | 0.7412 ± 0.0037 | 0.7354 ± 0.0032 | 0.7251 ± 0.0048 | 0.7320 ± 0.0043 | 0.7034 ± 0.0039 | <u>0.7684</u> ± 0.0027 | 0.7626 ± 0.0041 | 0.7002 ± 0.0040 | 0.7669 ± 0.0029        | **0.7702** ± 0.0025  |
| NDCG@5  | 0.6652 ± 0.0064 | 0.6072 ± 0.0081 | 0.6606 ± 0.0107 | 0.6564 ± 0.0090 | 0.6431 ± 0.0071 | 0.5825 ± 0.0136 | 0.6413 ± 0.0046 | <u>0.6923</u> ± 0.0038 | 0.6630 ± 0.0080 | 0.6298 ± 0.0050 | 0.6791 ± 0.0042        | **0.7135**\* ± 0.0039 |
| NDCG@10 | 0.6717 ± 0.0037 | 0.6203 ± 0.0031 | 0.6691 ± 0.0088 | 0.6622 ± 0.0069 | 0.6508 ± 0.0055 | 0.5981 ± 0.0060 | 0.6464 ± 0.0048 | 0.7008 ± 0.0052        | 0.6708 ± 0.0047 | 0.6416 ± 0.0070 | <u>0.7021</u> ± 0.0030 | **0.7192**\* ± 0.0043 |

KuaiSAR Src (999 negatives):
| Metric  | JSR             | SESRec                 | USER            | UnifiedSSR      | UniSAR          | MtlSAR               |
| ------- | --------------- | ---------------------- | --------------- | --------------- | --------------- | -------------------- |
| HR@1    | 0.0812 ± 0.0025 | <u>0.1478</u> ± 0.0032 | 0.0871 ± 0.0043 | 0.0986 ± 0.0045 | 0.0956 ± 0.0023 | **0.2065**\* ± 0.0037 |
| HR@5    | 0.1826 ± 0.0040 | <u>0.2961</u> ± 0.0038 | 0.2047 ± 0.0053 | 0.2009 ± 0.0059 | 0.2017 ± 0.0031 | **0.3720**\* ± 0.0052 |
| HR@10   | 0.2702 ± 0.0034 | <u>0.3869</u> ± 0.0034 | 0.2985 ± 0.0050 | 0.2881 ± 0.0055 | 0.2926 ± 0.0029 | **0.4511**\* ± 0.0043 |
| NDCG@5  | 0.1317 ± 0.0032 | <u>0.2221</u> ± 0.0040 | 0.1462 ± 0.0041 | 0.1504 ± 0.0055 | 0.1493 ± 0.0028 | **0.2869**\* ± 0.0039 |
| NDCG@10 | 0.1558 ± 0.0040 | <u>0.2462</u> ± 0.0041 | 0.1676 ± 0.0054 | 0.1750 ± 0.0058 | 0.1723 ± 0.0027 | **0.3164**\* ± 0.0038 |

JDsearch Src (999 negatives):
| Metric  | JSR             | SESRec                 | USER            | UnifiedSSR      | UniSAR          | MtlSAR               |
| ------- | --------------- | ---------------------- | --------------- | --------------- | --------------- | -------------------- |
| HR@1    | 0.4032 ± 0.0057 | <u>0.6831</u> ± 0.0050 | 0.4690 ± 0.0084 | 0.5927 ± 0.0087 | 0.6223 ± 0.0045 | **0.7005**\* ± 0.0046 |
| HR@5    | 0.6497 ± 0.0037 | <u>0.8110</u> ± 0.0032 | 0.7376 ± 0.0049 | 0.7318 ± 0.0046 | 0.7804 ± 0.0033 | **0.8562**\* ± 0.0025 |
| HR@10   | 0.7536 ± 0.0040 | <u>0.8754</u> ± 0.0030 | 0.8319 ± 0.0034 | 0.8066 ± 0.0035 | 0.8540 ± 0.0022 | **0.9004**\* ± 0.0024 |
| NDCG@5  | 0.5210 ± 0.0065 | <u>0.7630</u> ± 0.0038 | 0.5760 ± 0.0072 | 0.7014 ± 0.0064 | 0.7162 ± 0.0042 | **0.7691**\* ± 0.0044 |
| NDCG@10 | 0.5584 ± 0.0074 | <u>0.7788</u> ± 0.0039 | 0.6070 ± 0.0079 | 0.7248 ± 0.0068 | 0.7381 ± 0.0035 | **0.7864**\* ± 0.0034 |

KuaiSAR ablation (999 negatives):
| Model                                             |            Rec. NDCG@5 |           Rec. NDCG@10 |          Search NDCG@5 |         Search NDCG@10 |
| ------------------------------------------------- | ---------------------: | ---------------------: | ---------------------: | ---------------------: |
| MtlSAR                                            |    **0.0314** ± 0.0014 |   **0.0408**\* ± 0.0012 |   **0.2869**\* ± 0.0039 |   **0.3164**\* ± 0.0038 |
| w/o BAMHA                                         |        0.0282 ± 0.0014 |        0.0364 ± 0.0012 |        0.2271 ± 0.0053 |        0.2620 ± 0.0046 |
| w/o MCA                                           |        0.0287 ± 0.0010 |        0.0371 ± 0.0014 |        0.2315 ± 0.0065 |        0.2588 ± 0.0060 |
| w/o BAMHA & MCA                                   |        0.0254 ± 0.0011 |        0.0341 ± 0.0017 |        0.2246 ± 0.0071 |        0.2567 ± 0.0049 |
| w/o $\mathcal{L}_{macro}$                         |        0.0299 ± 0.0012 |        0.0382 ± 0.0017 | <u>0.2497</u> ± 0.0038 |        0.2743 ± 0.0065 |
| w/o $\mathcal{L}_{micro}$                         |        0.0293 ± 0.0012 | <u>0.0386</u> ± 0.0014 |        0.2255 ± 0.0041 |        0.2544 ± 0.0051 |
| w/o $\mathcal{L}_{macro}$ & $\mathcal{L}_{micro}$ |        0.0284 ± 0.0015 |        0.0374 ± 0.0012 |        0.2192 ± 0.0069 |        0.2503 ± 0.0044 |
| w/o TAAE (a)                                      | <u>0.0303</u> ± 0.0015 |        0.0380 ± 0.0013 |        0.2407 ± 0.0053 | <u>0.2756</u> ± 0.0041 |
| w/o TAAE (b)                                      |        0.0192 ± 0.0011 |        0.0255 ± 0.0019 |        0.2100 ± 0.0044 |        0.2412 ± 0.0065 |
| w/o TAAE (c)                                      |        0.0272 ± 0.0019 |        0.0359 ± 0.0014 |        0.2162 ± 0.0072 |        0.2481 ± 0.0045 |
| w/o Joint Training                                |        0.0281 ± 0.0017 |        0.0366 ± 0.0014 |        0.1825 ± 0.0061 |        0.2147 ± 0.0050 |



# Reviewer #1

We thank the reviewer for the careful and constructive comments. Below we address each point; new results were obtained under exactly the same data splits, negative sampling, and evaluation protocol as in the paper.

## Weak points

### W1. The novelty is somewhat incremental. Most components, including masked attention, cross-attention, contrastive learning, ordinal loss, and MoE-style experts, are established techniques. The main novelty is their integration around typed events, so stronger simple combination baselines are needed.
RW1. We agree that the individual modules are established, and we have added a Conventional Full Combination (CFC) baseline that composes them in the conventional way.

CFC uses the same data inputs, embeddings, history length, negative sampling, and evaluation protocol as MtlSAR. It combines event-type embeddings, two standard Transformer encoders, bidirectional cross-attention, candidate-aware attention pooling, a query-item InfoNCE loss, a search-recommendation InfoNCE loss, and a standard PLE layer. The query-item InfoNCE objective is the conventional counterpart to our graded relevance supervision and treats exposed-but-unclicked and random items as the same negative level. CFC excludes the three typed-event-specific mechanisms of MtlSAR: behavior-aware masked attention (BAMHA), ordinal query-item supervision, and target-aware expert routing.

On both datasets, CFC uses a one-layer, four-head Transformer with a hidden dimension of 128, four shared experts, four task-specific experts per task, a batch size of 1,024, and a recommendation-to-search loss ratio of 1:0.3. On JDsearch, the selected configuration uses a learning rate of 0.003 and loss weights of 0.1 and 0.01 for the query-item and search-recommendation InfoNCE objectives, respectively. On KuaiSAR, the learning rate is 0.01 and the two contrastive loss weights are 0.1 and $10^{-4}$. For CFC only, we refined the learning-rate search around the best point in the original grid and included 0.003 on JDsearch. We ran five random seeds and selected the checkpoint for each run using the mean of recommendation and search validation NDCG@5.

| Dataset | Scenario | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| KuaiSAR | Rec | $0.0647 \pm 0.0092$ | $0.2342 \pm 0.0052$ | $0.3793 \pm 0.0065$ | $0.1494 \pm 0.0063$ | $0.1960 \pm 0.0048$ |
| KuaiSAR | Src | $0.2958 \pm 0.0104$ | $0.5582 \pm 0.0084$ | $0.6839 \pm 0.0082$ | $0.4152 \pm 0.0048$ | $0.4661 \pm 0.0037$ |
| JDsearch | Rec | $0.7582 \pm 0.0028$ | $0.8930 \pm 0.0048$ | $0.9537 \pm 0.0103$ | $0.8396 \pm 0.0058$ | $0.8576 \pm 0.0138$ |
| JDsearch | Src | $0.8357 \pm 0.0026$ | $0.9627 \pm 0.0069$ | $0.9783 \pm 0.0052$ | $0.9160 \pm 0.0104$ | $0.9284 \pm 0.0068$ |

We do not claim that this configuration is an upper bound on conventional composition. It is a competitive reference point obtained under our tuning protocol. For recommendation on KuaiSAR, CFC outperforms 8 of the 11 published baselines on NDCG@5, with 0.1494 compared with 0.1441 for JSR, 0.1396 for SESRec, and 0.1411 for UnifiedSSR. For search, among the published baselines, only SESRec performs better on all five KuaiSAR metrics and on JDsearch HR@1 and NDCG@5. Its JDsearch NDCG@10 is higher than SESRec, while its HR@5 and HR@10 are lower than several baselines. Conventional composition still does not close the gap to MtlSAR. On KuaiSAR, MtlSAR raises NDCG@5 from 0.1494 to 0.1757 for recommendation and from 0.4152 to 0.5514 for search, corresponding to relative gains of 17.6% and 32.8%. This pattern is consistent with Table VI, where performance drops when BAMHA, ordinal query-item supervision, or target-aware expert routing is removed.

CFC also achieves a higher recommendation NDCG@5 than the `w/o TAAE (b)` ablation, 0.1494 versus 0.1066. This may seem counterintuitive because the ablation retains more MtlSAR components. The difference is the readout. `w/o TAAE (b)` sends the encoded history to standard PLE experts without candidate conditioning, whereas CFC applies candidate-aware attention pooling before PLE. This comparison supports the role of target conditioning at readout rather than the expert layer alone.

### W2. The evaluation is not convincing. Ranking is evaluated with only 99 sampled negatives rather than the full item catalog. Also, Table III reports 173,831 JDsearch users, while the text says only the first 50,000 users are used.
RW2. Please see our replies to D3 and D1.

### W3. Some assumptions and statistical details are unclear. The paper assumes exposed-but-unclicked items are always more relevant than random items, although exposure and position bias may violate this order. The main tables also report t-test significance without stating the number of runs or showing variance.
RW3. Please see our replies to D5 and D2.

## Detailed comments

### D1. Clarify whether Table III reports the original JDsearch statistics or the actual 50,000-user subset used in the experiments.
RD1. Table III reports statistics at the source-dataset level, i.e. before our preprocessing, for both KuaiSAR and JDsearch. We followed the convention of the most closely related S&R work (UniSAR [1], UnifiedSSR [3]), which also lists source-dataset statistics. We agree the text was not explicit about this and apologize for the confusion.

For efficiency, we use the first 50,000 users of JDsearch. The subset is a deterministic prefix of anonymized user IDs rather than a manually selected sample. This prevents us from tuning the user selection and makes the subset straightforward to reproduce. User subsampling is also used in this line of work; USER [5], for example, is built on a 100,000-user sample.

| Stage | Users | Items | Queries | Action-S | Action-R |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 50,000-user prefix before preprocessing | 50,000 | 4,843,600 | 49,384 | 1,747,826 | 4,957,309 |
| Experimental data after preprocessing | 35,461 | 233,341 | 35,008 | 784,243 | 1,644,824 |

The prefix retains 28.8% of users, 28.7% of recommendation actions, and 18.6% of search actions from the full dataset. It is slightly less search-dense per user, so it does not provide extra search evidence to any method.

### D2. Report the number of independent runs, mean, and standard deviation for Tables IV-VI.
RD2. All reported values are averages over five independent runs with five random seeds, using the same seeds for every method. This matches the protocol already stated in the Negative Transfer analysis ("... over five random trials"). The $t$-tests in Tables IV-VI compare MtlSAR with the second-best method over these five runs. We omitted standard deviations from the submitted tables only because the tables were already dense.

To keep the response focused, we do not reproduce the full per-cell tables here. In the revision, we will state that $n=5$ and report each result as mean $\pm$ standard deviation in Tables IV-VI. Across all entries in these tables, the largest standard deviation is 0.0147.

Table IV Rec KuaiSAR:
| Metric  | DIN             | SASRec          | CL4SRec                | NMTR            | NextIP          | EIDP            | JSR             | SESRec          | USER            | UnifiedSSR      | UniSAR                 | MtlSAR               |
| ------- | --------------- | --------------- | ---------------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---------------------- | -------------------- |
| HR@1    | 0.0612 ± 0.0023 | 0.0606 ± 0.0019 | 0.0665 ± 0.0026        | 0.0585 ± 0.0028 | 0.0538 ± 0.0020 | 0.0552 ± 0.0035 | 0.0609 ± 0.0026 | 0.0590 ± 0.0023 | 0.0670 ± 0.0020 | 0.0591 ± 0.0018 | <u>0.0689</u> ± 0.0015 | **0.0785*** ± 0.0026 |
| HR@5    | 0.2239 ± 0.0039 | 0.2262 ± 0.0026 | 0.2456 ± 0.0057        | 0.2174 ± 0.0033 | 0.2150 ± 0.0032 | 0.2161 ± 0.0073 | 0.2275 ± 0.0042 | 0.2205 ± 0.0043 | 0.2375 ± 0.0052 | 0.2234 ± 0.0033 | <u>0.2461</u> ± 0.0039 | **0.2720*** ± 0.0066 |
| HR@10   | 0.3592 ± 0.0031 | 0.3684 ± 0.0054 | <u>0.3940</u> ± 0.0078 | 0.3553 ± 0.0070 | 0.3574 ± 0.0040 | 0.3585 ± 0.0051 | 0.3683 ± 0.0035 | 0.3620 ± 0.0045 | 0.3770 ± 0.0086 | 0.3671 ± 0.0039 | 0.3938 ± 0.0029        | **0.4260*** ± 0.0043 |
| NDCG@5  | 0.1427 ± 0.0043 | 0.1432 ± 0.0027 | 0.1561 ± 0.0073        | 0.1378 ± 0.0035 | 0.1341 ± 0.0036 | 0.1353 ± 0.0030 | 0.1441 ± 0.0031 | 0.1396 ± 0.0044 | 0.1525 ± 0.0055 | 0.1411 ± 0.0057 | <u>0.1577</u> ± 0.0039 | **0.1757*** ± 0.0033 |
| NDCG@10 | 0.1861 ± 0.0025 | 0.1888 ± 0.0024 | 0.2038 ± 0.0081        | 0.1820 ± 0.0034 | 0.1798 ± 0.0051 | 0.1810 ± 0.0046 | 0.1893 ± 0.0050 | 0.1850 ± 0.0038 | 0.1974 ± 0.0032 | 0.1873 ± 0.0031 | <u>0.2052</u> ± 0.0037 | **0.2252*** ± 0.0030 |

Table IV Rec JDsearch:
| Metric  | DIN             | SASRec          | CL4SRec         | NMTR            | NextIP          | EIDP            | JSR             | SESRec                 | USER            | UnifiedSSR      | UniSAR                 | MtlSAR               |
| ------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | ---------------------- | --------------- | --------------- | ---------------------- | -------------------- |
| HR@1    | 0.7439 ± 0.0045 | 0.6176 ± 0.0081 | 0.7512 ± 0.0116 | 0.7306 ± 0.0107 | 0.7234 ± 0.0104 | 0.5727 ± 0.0070 | 0.6901 ± 0.0125 | 0.7737 ± 0.0060        | 0.7316 ± 0.0058 | 0.6900 ± 0.0068 | <u>0.7745</u> ± 0.0041 | **0.8196*** ± 0.0038 |
| HR@5    | 0.9142 ± 0.0040 | 0.8813 ± 0.0047 | 0.9050 ± 0.0051 | 0.9022 ± 0.0055 | 0.8780 ± 0.0051 | 0.8779 ± 0.0048 | 0.8694 ± 0.0043 | <u>0.9323</u> ± 0.0046 | 0.9203 ± 0.0035 | 0.8619 ± 0.0049 | 0.9300 ± 0.0046        | **0.9441*** ± 0.0034 |
| HR@10   | 0.9424 ± 0.0042 | 0.9165 ± 0.0038 | 0.9314 ± 0.0034 | 0.9292 ± 0.0038 | 0.9075 ± 0.0050 | 0.9168 ± 0.0046 | 0.8992 ± 0.0045 | <u>0.9541</u> ± 0.0034 | 0.9473 ± 0.0055 | 0.8974 ± 0.0044 | 0.9512 ± 0.0041        | **0.9616** ± 0.0026 |
| NDCG@5  | 0.8394 ± 0.0074 | 0.7659 ± 0.0085 | 0.8384 ± 0.0119 | 0.8285 ± 0.0094 | 0.8095 ± 0.0094 | 0.7429 ± 0.0147 | 0.7919 ± 0.0047 | <u>0.8648</u> ± 0.0037 | 0.8391 ± 0.0095 | 0.7868 ± 0.0051 | 0.8638 ± 0.0048        | **0.8912*** ± 0.0046 |
| NDCG@10 | 0.8486 ± 0.0037 | 0.7774 ± 0.0041 | 0.8469 ± 0.0086 | 0.8373 ± 0.0079 | 0.8192 ± 0.0051 | 0.7557 ± 0.0060 | 0.8016 ± 0.0048 | <u>0.8719</u> ± 0.0061 | 0.8480 ± 0.0047 | 0.7983 ± 0.0071 | 0.8706 ± 0.0031        | **0.8969*** ± 0.0058 |

Table V Src KuaiSAR:
| Metric  | JSR             | SESRec                 | USER            | UnifiedSSR      | UniSAR          | MtlSAR               |
| ------- | --------------- | ---------------------- | --------------- | --------------- | --------------- | -------------------- |
| HR@1    | 0.2267 ± 0.0039 | <u>0.3256</u> ± 0.0043 | 0.2363 ± 0.0089 | 0.2541 ± 0.0102 | 0.2500 ± 0.0038 | **0.3953*** ± 0.0044 |
| HR@5    | 0.5248 ± 0.0082 | <u>0.6303</u> ± 0.0042 | 0.5501 ± 0.0097 | 0.5441 ± 0.0117 | 0.5445 ± 0.0047 | **0.6869*** ± 0.0075 |
| HR@10   | 0.6680 ± 0.0043 | <u>0.7472</u> ± 0.0039 | 0.6879 ± 0.0066 | 0.6806 ± 0.0087 | 0.6839 ± 0.0035 | **0.7830*** ± 0.0037 |
| NDCG@5  | 0.3823 ± 0.0045 | <u>0.4862</u> ± 0.0062 | 0.4005 ± 0.0051 | 0.4063 ± 0.0111 | 0.4040 ± 0.0041 | **0.5514*** ± 0.0048 |
| NDCG@10 | 0.4289 ± 0.0066 | <u>0.5241</u> ± 0.0056 | 0.4451 ± 0.0112 | 0.4504 ± 0.0117 | 0.4493 ± 0.0037 | **0.5827*** ± 0.0047 |

Table V Src JDsearch:
| Metric  | JSR             | SESRec                 | USER            | UnifiedSSR      | UniSAR          | MtlSAR               |
| ------- | --------------- | ---------------------- | --------------- | --------------- | --------------- | -------------------- |
| HR@1    | 0.6724 ± 0.0053 | <u>0.8450</u> ± 0.0051 | 0.7212 ± 0.0095 | 0.7982 ± 0.0099 | 0.8128 ± 0.0040 | **0.8553*** ± 0.0046 |
| HR@5    | 0.9447 ± 0.0028 | <u>0.9769</u> ± 0.0023 | 0.9635 ± 0.0038 | 0.9628 ± 0.0032 | 0.9715 ± 0.0028 | **0.9838** ± 0.0013 |
| HR@10   | 0.9726 ± 0.0031 | <u>0.9883</u> ± 0.0022 | 0.9834 ± 0.0017 | 0.9798 ± 0.0021 | 0.9856 ± 0.0011 | **0.9909** ± 0.0017  |
| NDCG@5  | 0.8271 ± 0.0070 | <u>0.9213</u> ± 0.0029 | 0.8605 ± 0.0080 | 0.8929 ± 0.0054 | 0.9046 ± 0.0043 | **0.9309*** ± 0.0047 |
| NDCG@10 | 0.8363 ± 0.0092 | <u>0.9250</u> ± 0.0041 | 0.8670 ± 0.0083 | 0.8984 ± 0.0084 | 0.9092 ± 0.0034 | **0.9333*** ± 0.0024 |

Table VI:

| Model                                             | NDCG@5                 | NDCG@10                | NDCG@5                 | NDCG@10                |
| ------------------------------------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| MtlSAR                                            | **0.1757*** ± 0.0033   | **0.2252*** ± 0.0030   | **0.5514*** ± 0.0048   | **0.5827*** ± 0.0047   |
| w/o BAMHA                                         | 0.1568 ± 0.0043        | 0.2037 ± 0.0029        | 0.4387 ± 0.0073        | 0.4809 ± 0.0066        |
| w/o MCA                                           | 0.1587 ± 0.0026        | 0.2056 ± 0.0040        | 0.4394 ± 0.0107        | 0.4779 ± 0.0098        |
| w/o BAMHA & MCA                                   | 0.1430 ± 0.0027        | 0.1884 ± 0.0055        | 0.4341 ± 0.0116        | 0.4731 ± 0.0063        |
| w/o $\mathcal{L}_{macro}$                         | <u>0.1687</u> ± 0.0031 | <u>0.2124</u> ± 0.0055 | <u>0.4743</u> ± 0.0048 | <u>0.5082</u> ± 0.0113 |
| w/o $\mathcal{L}_{micro}$                         | 0.1638 ± 0.0039        | 0.2111 ± 0.0043        | 0.4320 ± 0.0050        | 0.4727 ± 0.0067        |
| w/o $\mathcal{L}*{macro}$ & $\mathcal{L}*{micro}$ | 0.1598 ± 0.0040        | 0.2075 ± 0.0037        | 0.4190 ± 0.0115        | 0.4613 ± 0.0056        |
| w/o TAAE (a)                                      | 0.1643 ± 0.0046        | 0.2113 ± 0.0034        | 0.4576 ± 0.0084        | 0.4990 ± 0.0048        |
| w/o TAAE (b)                                      | 0.1066 ± 0.0032        | 0.1439 ± 0.0073        | 0.4061 ± 0.0060        | 0.4480 ± 0.0109        |
| w/o TAAE (c)                                      | 0.1534 ± 0.0068        | 0.1987 ± 0.0036        | 0.4108 ± 0.0120        | 0.4583 ± 0.0057        |
| w/o Joint Training                                | 0.1579 ± 0.0059        | 0.2035 ± 0.0038        | 0.3524 ± 0.0089        | 0.3950 ± 0.0071        |


### D3. A full-ranking evaluation, or at least evaluation with more sampled negatives, would make the results more convincing.
RD3. We used 99 sampled negatives to match the evaluation protocol of the baselines. SESRec [4], UnifiedSSR [3], UniSAR [1], and the recent LCR-SER [2] all pair the ground-truth item with 99 randomly sampled negatives; among the S&R baselines, only JSR [11] differs. USER [5] uses a different but not directly comparable protocol: 9 negatives selected by popularity and topic similarity for recommendation, and re-ranking of the real top-20 impression list for search. The 99-negative protocol is also widely used in sequential recommendation [6, 7, 8, 9, 10]. Using a different candidate construction for MtlSAR would prevent a fair comparison with the baselines in Tables IV-V.

There is also a cost consideration specific to our scorer. MtlSAR is not a dot-product model. Its score uses target-aware attention and task-specific prediction layers, as shown in Eqs. (13)-(17). The shared history encoding can be reused, but the target-dependent readout must be recomputed for each candidate. Full-catalog scoring therefore requires target-dependent computation over 233,341 items on the JDsearch subset and the full item set on KuaiSAR for each test instance. Repeating this evaluation for every method and seed is expensive.

We agree, nonetheless, that a harder candidate set is informative. We therefore repeated the evaluation with 999 randomly sampled negatives for all methods. Because of space limits, we placed the full result table in the README of our GitHub repository rather than in the paper. The relative ordering of the methods and all conclusions drawn in the paper are unchanged.

### D4. The modifications made to the baselines should be described more precisely. Adding search and multi-behavior inputs may substantially change the original models.
RD4. We apologize for the brevity of the original description. The modifications were deliberately confined to the input history representation, so that the baselines' own modeling contributions remain intact.

For all baselines, we keep their original interest-aggregation, sequence-encoding, contrastive, or multi-behavior backbones and their original recommendation prediction objective, and only extend the historical event representation in a uniform way:

1. Behaviors on a recommended item are encoded as a multi-hot behavior vector multiplied by a learnable behavior embedding matrix; the resulting behavior representation is added to the item representation. The behavior sets are {click, like, follow, forward} for KuaiSAR and {click, order, add-to-cart, follow} for JDsearch.
2. Each historical search session is represented as the sum of the query embedding and the mean embedding of the items clicked under that query. Queries are obtained by mean-pooling word embeddings followed by a linear projection in the baseline input adapter; the search-session encoding is implemented in `inputs.py` of our repository.
3. Recommendation and search events are merged into one chronologically ordered history, which is then fed into the baseline's own encoder.

Importantly, these baselines are still trained with recommendation labels and their original losses only, and the current target query never enters the recommendation prediction layer. The enhancement therefore acts on historical evidence and does not turn these models into joint S&R models.

For SESRec, we keep its dual-sequence Transformer, cross-sequence interaction and contrastive modules unchanged. Beyond the input alignment above, the only addition is a search prediction head placed in parallel with the recommendation head so that search ranking is supported. The recommendation head takes $[h_u;e_u;e_i]$, and the search head additionally concatenates the current query, $[h_u;e_u;e_q;e_i]$; both heads are MLPs of identical structure with independent parameters, each trained with binary cross-entropy. Keeping SESRec's original contrastive and query-item alignment terms, the outer joint objective is $\mathcal{L}=\mathcal{L}_{\text{rec}}+\lambda_{\text{src}}\mathcal{L}_{\text{src}}$.

All baselines use the same data splits, negative sampling, and evaluation protocol, and have access to the same raw behavior and search information as MtlSAR, which removes comparison bias caused by unequal input information.

### D5. The assumption that clicked, exposed-but-unclicked, and random items always form a strict relevance order needs more discussion or an empirical check.
RD5. We agree that "strict order" would be too strong, and this is not what we assume. The paper states the assumption in graded, population-level terms: a clicked item "indicates high relevance", an exposed-but-unclicked item "usually implies partial relevance", and a randomly sampled item "is typically irrelevant". Correspondingly, Eq. (10) is a probabilistic ordinal likelihood with learnable thresholds, not a hard constraint: a violated instance contributes a finite penalty and behaves like label noise, rather than making the objective infeasible. The assumption is used as a soft prior on the expected order, not as a per-instance guarantee.

The three-level ordering is also not new to this work. It is a natural extension of the two-level assumption of BPR [12] to multi-feedback settings; the ordering "strong interaction > weak/intermediate feedback > unobserved" has been adopted in impression-aware recommendation [13] and shown to be effective by the view-enhanced BPR sampler [14]. DFN [15] likewise treats clicks and dislikes as strong signals while regarding exposed-but-unclicked feedback as weaker and noisier. For the search setting specifically, [19] show that, within the same search context, preferences between a clicked item and an examined-but-skipped item are relatively reliable, even though absolute click-based relevance is biased.

We also agree that the order can be violated for individual examples. An exposed item may be relevant but never examined, its non-click may reflect position bias [16, 17], and a click may be induced or noisy [18]. Fig. 4 is consistent with a graded interpretation. Adding $\mathcal{L}_{\text{micro}}$ clearly separates random negatives, while positives and weak negatives retain a smaller margin. This pattern is consistent with weak negatives being partly relevant and partly noisy. A position-controlled test such as "click > skip-above" would be more direct, but neither KuaiSAR nor JDsearch records within-impression positions or exposure order. We therefore cannot condition the analysis on examination.

### D6. Explain how the model guarantees that the two ordinal thresholds remain correctly ordered during training.
RD6. The ordering is guaranteed by construction through a reparameterization rather than by a penalty. The lower threshold is defined as $\theta_1=m_1$ and the upper threshold as $\theta_2=m_1+\mathrm{Softplus}(m_2)$, where $m_1$ and $m_2$ are the underlying learnable parameters. Since $\mathrm{Softplus}(m_2) > 0$ for any finite $m_2$, $\theta_2>\theta_1$ holds at every training step, so no sorting, clipping, or additional constraint loss is required, and both parameters are learned jointly by backpropagation. The implementation is the `OrdinalLogitRegression` class in `layers.py` of our repository.

# Reviewer #2

We thank the reviewer for the detailed and technically precise comments. Several of them point at genuine gaps in the description of our implementation, and we address each below. All new measurements were obtained under the same data splits, negative sampling, and evaluation protocol as in the paper.

## Weak points

### W1. Most components build on established techniques, and the main novelty lies in their integration under the typed-event framework.
RW1. We agree with this characterization and have narrowed our claim accordingly: what we propose is the typed-event formulation and the modeling that this formulation makes possible, not the individual modules.

To test whether a conventional composition of the same modules is sufficient, we added a **Conventional Full Combination (CFC)** baseline. CFC uses the same inputs, embeddings, history length, negative sampling, and evaluation protocol as MtlSAR. It combines event-type embeddings, two standard Transformer encoders, bidirectional cross-attention, candidate-aware attention pooling, a query-item InfoNCE loss, a search-recommendation InfoNCE loss, and a standard PLE layer. The query-item InfoNCE loss treats exposed-but-unclicked and random items as the same negative level. CFC excludes the three typed-event-specific mechanisms: behavior-aware masked attention (BAMHA), ordinal query-item supervision, and target-aware expert routing.

| Dataset | Scenario | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| KuaiSAR | Rec | $0.0647 \pm 0.0092$ | $0.2342 \pm 0.0052$ | $0.3793 \pm 0.0065$ | $0.1494 \pm 0.0063$ | $0.1960 \pm 0.0048$ |
| KuaiSAR | Src | $0.2958 \pm 0.0104$ | $0.5582 \pm 0.0084$ | $0.6839 \pm 0.0082$ | $0.4152 \pm 0.0048$ | $0.4661 \pm 0.0037$ |
| JDsearch | Rec | $0.7582 \pm 0.0028$ | $0.8930 \pm 0.0048$ | $0.9537 \pm 0.0103$ | $0.8396 \pm 0.0058$ | $0.8576 \pm 0.0138$ |
| JDsearch | Src | $0.8357 \pm 0.0026$ | $0.9627 \pm 0.0069$ | $0.9783 \pm 0.0052$ | $0.9160 \pm 0.0104$ | $0.9284 \pm 0.0068$ |

CFC is a competitive reference point. For recommendation on KuaiSAR, it outperforms 8 of the 11 published baselines on NDCG@5. For search, among the published baselines, only SESRec performs better on all five KuaiSAR metrics and on JDsearch HR@1 and NDCG@5. Conventional composition still does not close the gap to MtlSAR. On KuaiSAR, MtlSAR raises NDCG@5 from 0.1494 to 0.1757 for recommendation and from 0.4152 to 0.5514 for search, corresponding to relative gains of 17.6% and 32.8%. This is consistent with Table VI, where removing the typed-event-specific mechanisms reduces performance.

### W2. Equation (13) does not appear to include expert-specific attention parameters, making it unclear whether different experts actually attend to different history positions.
RW2. Please see D1 for the exact parameter-sharing scope and the corresponding revision.

### W3. The paper does not explain how the ordering constraint $\theta_1<\theta_2$ is enforced or how numerical stability is handled.
RW3. Please see D2 for the parameterization, initialization, optimization, and the numerical-stability measures.

### W4. The use of only 99 random negatives may not sufficiently reflect realistic large-scale retrieval or ranking conditions.
RW4. Two clarifications. First, MtlSAR targets the **re-ranking** stage rather than first-stage retrieval. It scores a candidate set produced upstream, which is also the deployment stage of the joint S&R models in our comparison. Some baselines use factorized scorers and can also support full-catalog retrieval, so we do not present candidate-set evaluation as equivalent to full retrieval.

Second, we used 99 sampled negatives for comparability with the baselines. SESRec [4], UnifiedSSR [3], UniSAR [1], and the recent LCR-SER [2] use the same protocol; among the S&R baselines, only JSR [11] differs. USER [5] uses a different, not directly comparable protocol: 9 negatives selected by popularity and topic similarity for recommendation, and re-ranking of the real top-20 impression list for search. The 99-negative protocol is also widely used in sequential recommendation [6, 7, 8, 9, 10]. Using a different candidate construction for MtlSAR would prevent a fair comparison in Tables IV-V.

There is also a cost consideration specific to our scorer. MtlSAR is not a dot-product model, and its score uses the target-dependent layers in Eqs. (13)-(17). The shared history encoding is computed once, but target-aware attention and the prediction layers must be recomputed for each candidate. Full-catalog scoring therefore grows linearly with the catalog size and is costly when repeated for every method and seed.

We nevertheless agree that a harder candidate set is informative and repeated the evaluation with **999 randomly sampled negatives** for all methods. Because of space limits, the full table is in the README of our GitHub repository. The relative ordering of methods and all conclusions in the paper are unchanged.

### W5. The baseline modifications, repeated-run statistics, and computational costs are not described in enough detail.
RW5. (a) Baseline modifications. We limit the changes to the input history representation so that each baseline retains its original modeling contribution. For the recommendation baselines, we keep the original backbone and recommendation objective and make three uniform input changes:

1. Behaviors on a recommended item are encoded by multiplying a multi-hot behavior vector with a learnable behavior embedding matrix and adding the result to the item representation. The behavior sets are {click, like, follow, forward} for KuaiSAR and {click, order, add-to-cart, follow} for JDsearch.
2. Each historical search session is represented by the query embedding plus the mean embedding of the items clicked under that query.
3. Recommendation and search events are merged into one chronological history and passed to the baseline's original encoder.

These baselines are still trained only with recommendation labels and their original losses. The current target query never enters the recommendation prediction layer, so these changes do not turn them into joint S&R models.

For SESRec, we keep the dual-sequence Transformer, cross-sequence interaction, and contrastive modules unchanged. We add only a parallel search head. The recommendation head receives $[h_u;e_u;e_i]$, while the search head receives $[h_u;e_u;e_q;e_i]$. Both heads are MLPs with the same structure and separate parameters, and both use binary cross-entropy. The outer objective is $\mathcal{L}=\mathcal{L}_{\text{rec}}+\lambda_{\text{src}}\mathcal{L}_{\text{src}}$ in addition to SESRec's original terms.

(b) Repeated-run statistics. All reported values are averages over five independent runs with five random seeds, using the same seeds for every method. This is consistent with the protocol already stated in the Negative Transfer analysis. The $t$-tests in Tables IV-VI compare MtlSAR with the second-best method over these five runs. Standard deviations were omitted only for space; the maximum standard deviation across all entries is 0.0147. We will add $n=5$ and mean $\pm$ standard deviation to the revised tables.

(c) Computational cost. Please see D5.

## Detailed comments

### D1. Clarify whether each target-aware expert has independent attention projections or parameters. If attention is shared within an expert group, the description should explain how expert diversity is achieved.
RD1. We thank the reviewer for catching this; the current notation in Eq. (13) is misleading and we will correct it.

In our implementation, experts do not have independent Q/K/V projections. Target-aware attentive aggregation is performed once for each expert group $g\in\{\text{rec},\text{src},\text{share}\}$. All experts within a group receive the same aggregated context vector, after which each expert applies its own non-shared MLP. Expert diversity therefore comes from three sources:

1. Independent expert transformations. Each expert has its own Swish MLP parameters, so the same context is mapped into different subspaces.
2. Group-specific input feature gating (Eq. (12)). The shared group and the two task-specific groups use different gating parameters, so the three groups already attend over different feature subspaces of $H_u$.
3. Sample-dependent dynamic routing. Recommendation and search use separate target-conditioned PLE gating networks (Eq. (15)), so the combination of shared and task-specific experts varies per sample and per task.

The analysis in the paper provides supporting evidence. Fig. 6 shows that the experts retain distinct, nonzero average weights and exhibit different task preferences. Fig. 7 shows that the gating-weight distributions for recommendation and search form distinct clusters in the t-SNE space.

### D2. Describe the parameterization, initialization, and optimization of the ordinal thresholds, including how their ordering is guaranteed.
RD2. **Parameterization and ordering guarantee.** The ordering is enforced by construction rather than by a penalty:

$$
\theta_1=m_1, \qquad \theta_2=m_1+\mathrm{Softplus}(m_2),
$$

where $m_1$ and $m_2$ are unconstrained learnable scalars. Because $\mathrm{Softplus}(m_2)>0$ for every finite $m_2$, $\theta_2>\theta_1$ holds at every training step. No projection, sorting, or constraint loss is needed. The implementation is the `OrdinalLogitRegression` class in `layers.py`.

**Similarity scale.** The similarity in Eqs. (9)-(10) is computed from $\ell_2$-normalized query and item representations. It is therefore cosine similarity bounded in $[-1,1]$. Section V-D1 already reports cosine similarities between queries and item types, but Eq. (9) does not state the normalization explicitly. We will add this detail in the revision.

**Initialization.** We initialize $m_1=0.2$ and $m_2=-2.0$. This gives $\theta_1=0.2$ and $\theta_2=0.2+\mathrm{Softplus}(-2.0)\approx0.3269$, so the initial band width is about $0.127$. Since $\sigma'(z)\le 1/4$, the middle-band probability is at most $(\theta_2-\theta_1)/4\approx0.03$ at initialization. The weak-negative likelihood therefore provides a direct gradient for widening the band during training.

**Optimization.** Let $p$ denote a clicked item, $wn$ an exposed-but-unclicked item, and $rn$ a random negative. For $s=\mathrm{sem\_sim}(e_q,e_x)$, the three level probabilities are

$$
\begin{aligned}
P(p\mid s) &= \sigma(s-\theta_2), \\
P(wn\mid s) &= \sigma(\theta_2-s)-\sigma(\theta_1-s), \\
P(rn\mid s) &= \sigma(\theta_1-s).
\end{aligned}
$$

Equation (10) is the negative log-likelihood of these probabilities. We optimize it jointly with all other parameters by Adam within $\mathcal{L}_{\text{total}}$ in Eq. (20). The two threshold scalars are excluded from $\Theta$ in the $\ell_2$ regularization term of Eq. (20), so no weight decay is applied to them.

**Numerical stability.** First, $\ell_2$ normalization bounds $s$ in $[-1,1]$, which keeps the sigmoid inputs in a stable range. Second, Softplus enforces a positive band width. Third, the middle-band probability is clamped to at least $\epsilon=10^{-8}$ before taking the logarithm, which prevents $\log(0)$ after floating-point cancellation. Finally, padded weak negatives are masked out of the loss.

### D3. Explain how search events with no clicked items are represented, since the current formulation relies on mean pooling over the clicked-item set.
RD3. Search events with an empty clicked-item set are removed during preprocessing, and we apologize for not documenting this step. Their share is very small: 46,855 of 3,171,231 search events (1.48%) on KuaiSAR and 8,474 of 1,747,826 (0.48%) on JDsearch.

The filter follows directly from our typed-event schema rather than being an ad-hoc choice. Section III states that a search event requires a query and clicked items, and Eq. (3) defines its representation as $e_q+\mathrm{Mean}(C_q)$. The same requirement is implicit in closely related work that constructs search events from a query and its clicked items [1, 4] and, to our understanding, in [2, 3, 5] as well. The filter affects only history construction and training samples. Evaluation instances are click-based by definition, so no test example is discarded. We apply the same filter to every baseline.

Extending the model to zero-click search events would be straightforward (e.g. using $e_q$ alone, or $e_q$ plus the mean embedding of the exposed items $E_q$), and given the proportions above we do not expect it to change the conclusions. We also note as a limitation that abandoned searches may themselves carry a dissatisfaction signal that our current schema does not model.

### D4. Discuss the limitations of treating exposed-but-unclicked items as weak negatives, especially under position and examination bias.
RD4. We agree with the concern and clarify that we do not assume a strict order for every example. The paper uses graded, population-level language: a clicked item "indicates high relevance", an exposed-but-unclicked item "usually implies partial relevance", and a random item "is typically irrelevant". Equation (10) implements this assumption as a probabilistic ordinal likelihood with learnable thresholds, not as a hard constraint. A violation contributes a finite penalty and behaves like label noise rather than making the objective infeasible. The selected weight, $\lambda_{\text{micro}}=10^{-3}$ in Section V-D1, is small, so this loss acts as a mild regularizer on the semantic space.

The three-level ordering is also not new. It extends the two-level assumption of BPR [12] to multi-feedback settings. Impression-aware recommendation [13] adopts the same strong-interaction, weak-feedback, and unobserved ordering, and the view-enhanced BPR sampler [14] shows that it can be effective. DFN [15] likewise treats clicks and dislikes as strong signals while regarding exposed-but-unclicked feedback as weaker and noisier. For search, [19] shows that preferences between a clicked item and an examined-but-skipped item in the same context are relatively reliable, even though absolute click-based relevance is biased.

That said, we agree that the order can be violated for individual examples. An exposed item may be relevant but never examined, its non-click may reflect position bias [16, 17], and clicks may be induced or noisy [18]. Two points reduce, but do not remove, this concern. First, OLR places weak negatives in the band between $\theta_1$ and $\theta_2$ instead of pushing them away from the query. By contrast, when an InfoNCE objective treats exposed-but-unclicked and random items as the same negative level, it does not preserve this distinction. Our formulation may therefore be less sensitive when some weak negatives are actually relevant. Second, Fig. 4 supports a graded interpretation: adding $\mathcal{L}_{\text{micro}}$ clearly separates random negatives, while positives and weak negatives retain a smaller margin. This pattern is consistent with weak negatives being partly relevant and partly noisy.

Finally, an examination-controlled check in the spirit of "click > skip-above" would be the most direct empirical test, but neither KuaiSAR nor JDsearch records within-impression positions or exposure order, so examination cannot be conditioned on in these datasets.

### D5. Report model efficiency, including parameter count, training time, memory usage, and inference latency relative to the strongest baselines.
RD5. We report MtlSAR against UniSAR and SESRec, the two baselines with the best overall search-and-recommendation performance. All measurements use a single NVIDIA L20 (46 GB) in FP32, with a training batch size of 1,024.

**Parameters.** After shared storage is counted once, the parameter counts for MtlSAR, UniSAR, and SESRec are 20.60, 17.52, and 16.85 M on JDsearch and 44.91, 41.82, and 41.15 M on KuaiSAR, respectively. MtlSAR has between 7.4% and 22.3% more parameters than the baselines but remains in the same order of magnitude. Shared input representations account for 76.3% of its parameters on JDsearch and 89.1% on KuaiSAR. For clarity, the item lookup table stores a 32-dimensional ID embedding, which is concatenated with category and shop embeddings and projected to the $d=128$ event dimension. All three models use the same input configuration.

**Training.** Per-epoch time is 16.56 / 11.92 / 7.74 min on JDsearch, with peak training memory of 6.14 / 5.46 / 3.31 GiB. On KuaiSAR, per-epoch time is 6.91 / 6.72 / 5.22 min and peak training memory is 4.55 / 3.78 / 2.40 GiB. Total training time is 3.99 / 3.11 / 2.03 h on JDsearch and 3.36 / 2.75 / 3.42 h on KuaiSAR. On JDsearch, the three models converge after a comparable number of epochs. The longer total time of MtlSAR there is therefore explained mainly by its higher per-epoch cost.

**Inference memory.** Peak inference memory for MtlSAR is 1.62 GiB on KuaiSAR and 1.69 GiB on JDsearch. These values are lower than UniSAR's 2.96 and 3.03 GiB, but higher than SESRec's 1.11 and 1.18 GiB.

**Inference latency.** We measured pure GPU inference on the same device with a batch size of 512. We warmed up 20 batches per task and timed 100 batches with CUDA events. Data loading, host-to-device transfer, and metric computation were excluded. On JDsearch, MtlSAR uses 0.0961 / 0.0955 ms per query and reaches 10,404 / 10,475 QPS for recommendation / search. The corresponding results are 0.0625 / 0.0678 ms and 16,004 / 14,745 QPS for SESRec, and 0.1136 / 0.1272 ms and 8,799 / 7,859 QPS for UniSAR. On KuaiSAR, MtlSAR uses 0.0838 / 0.0859 ms and reaches 11,927 / 11,641 QPS. SESRec uses 0.0661 / 0.0620 ms and reaches 15,137 / 16,135 QPS, while UniSAR uses 0.1107 / 0.1129 ms and reaches 9,031 / 8,856 QPS. MtlSAR therefore has lower latency and higher throughput than UniSAR, but remains slower than the lighter SESRec.

**Complexity.** Let $M$ be the batch size, $A$ the number of attention layers, $L$ the total history length, $|\mathcal{B}|$ the number of behavior types, $N$ the number of candidates, $T$ the number of tasks, and $E_s$ and $E_t$ the numbers of shared and task-specific experts:

- MtlSAR: $O\!\left(MA(L^2d+Ld^2)+ML^2|\mathcal{B}|+M(T+1)NLd+M(E_s+TE_t)Nd^2\right)$
- UniSAR: $O\!\left(M[A(L^2d+Ld^2)+L|\mathcal{B}|d+NLd^2]\right)$
- SESRec: $O\!\left(M[A(L^2d+Ld^2)+L^2d+L|\mathcal{B}|d+NLd]\right)$

All three models contain an $O(L^2)$ self-attention term and grow linearly with $|\mathcal{B}|$. Their candidate-dependent terms differ. SESRec uses $O(MNLd)$ matching, while UniSAR uses an $O(MNLd^2)$ candidate-aware path. Under our configuration, $L=60$ (30 events per scenario), $d=128$, $|\mathcal{B}|=4$, $T=2$, and $E_s=E_t=4$, with one attention layer. The candidate-dependent cost per candidate is about $2.2\times10^5$ for MtlSAR and $9.8\times10^5$ for UniSAR, or about $0.22\times$. This difference is one of the main reasons that MtlSAR has lower peak inference memory than UniSAR despite having more parameters. The remaining memory difference comes from fixed costs. In MtlSAR, target attention aggregates the history before the experts, so the expert count does not multiply the $L^2$ term. Adding a behavior type also does not require another full encoder. All three models use in-batch contrastive objectives during training. For MtlSAR, $\mathcal{L}_{\text{macro}}$ adds $O(M^2d)$ time and $O(M^2)$ memory during training only.

### D6. Include variance or confidence intervals for the gradient cosine analysis, since the reported values are extremely close to zero and may not support strong conclusions based only on their signs.
RD6. We omitted the error bars in Fig. 5 only for readability. The full statistics over the same five runs are:

| Model | Mean $\pm$ Std. | 95% CI |
| :--- | ---: | ---: |
| MtlSAR          |   $(6.39 \pm 2.13) \times 10^{-5}$ |    $[3.75, 9.03] \times 10^{-5}$ |
| UniSAR          | $(-0.275 \pm 1.07) \times 10^{-5}$ |   $[-1.60, 1.05] \times 10^{-5}$ |
| MtlSAR w/o TAAE |  $(-8.90 \pm 2.84) \times 10^{-5}$ | $[-12.43, -5.37] \times 10^{-5}$ |

Intervals are computed as $\text{mean} \pm t_{0.975,4} \cdot s/\sqrt{5}$ with $t_{0.975,4} = 2.776$ and $n = 5$. Two-sample $t$-tests ($df = 8$) give $t \approx 6.3$ for MtlSAR vs. UniSAR and $t \approx 9.6$ for MtlSAR vs. MtlSAR w/o TAAE, both with $p < 0.001$.

We agree that the sign alone is a weak basis for a claim. The confidence intervals give a clearer result. UniSAR's interval spans zero and is statistically indistinguishable from no alignment. MtlSAR's interval lies entirely above zero, while the interval for MtlSAR w/o TAAE lies entirely below zero. The conclusion is supported by this separation rather than by the absolute magnitude alone.

The cosine is computed between gradients of the shared item embedding layer, which lies in a space with millions of dimensions. For random directions in $D$ dimensions, the cosine scale is $O(1/\sqrt{D})$. The observed values are below that scale, so their small absolute magnitude is not surprising. Our conclusion does not rely on the magnitude alone. It relies on the confidence intervals, the consistency across seeds, and the separation among the three configurations.

# Reviewer #3

We thank the reviewer for the focused and constructive comments. W1-W3 restate D1-D3, so we answer each concern once, under the corresponding detailed comment. All new results use the same data splits, negative sampling, and evaluation protocol as in the paper, averaged over five independent runs with five random seeds.

## Weak points

### W1: The overall technical novelty is somewhat incremental.
RW1. Please see D1.

### W2: Efficiency, scalability, and deployment costs are not evaluated.
RW2. Please see D2.

### W3: The experimental protocol based on 99 sampled negatives is insufficient for strong ranking claims.
RW3. Please see D3.

## Detailed comments

### D1: The overall technical novelty is somewhat incremental. Although the typed-event framing is meaningful, most architectural components are adaptations or combinations of established techniques: masked multi-head attention for behavior-aware modeling, bidirectional cross-attention for scenario fusion, InfoNCE for representation alignment, ... The main novelty appears to lie in assembling these components for unified search and recommendation rather than introducing a fundamentally new learning paradigm. The paper should more precisely distinguish its technical contributions from UniSAR, UnifiedSSR, SESRec, multi-behavior Transformers, and target-aware MoE models.
RD1. We agree that the individual modules are established, and we have narrowed our claim accordingly: we do not propose a new learning paradigm. What we claim is the typed-event schema as the modeling unit for unified S&R logs, together with the three mechanisms that this unit makes possible.

Is conventional composition already sufficient? To test this directly, we added a Conventional Full Combination (CFC) baseline. CFC uses the same inputs, embeddings, history length, negative sampling, and evaluation protocol as MtlSAR. It combines event-type embeddings, two standard Transformer encoders, bidirectional cross-attention, candidate-aware attention pooling, a query-item InfoNCE loss, a search-recommendation InfoNCE loss, and a standard PLE layer. It excludes the same three typed-event-specific mechanisms used throughout this response: behavior-aware masked attention (BAMHA), ordinal query-item supervision, and target-aware expert routing.

| Dataset | Scenario | HR@1 | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| KuaiSAR  | Rec      | $0.0647 \pm 0.0092$ | $0.2342 \pm 0.0052$ | $0.3793 \pm 0.0065$ | $0.1494 \pm 0.0063$ | $0.1960 \pm 0.0048$ |
| KuaiSAR  | Src      | $0.2958 \pm 0.0104$ | $0.5582 \pm 0.0084$ | $0.6839 \pm 0.0082$ | $0.4152 \pm 0.0048$ | $0.4661 \pm 0.0037$ |
| JDsearch | Rec      | $0.7582 \pm 0.0028$ | $0.8930 \pm 0.0048$ | $0.9537 \pm 0.0103$ | $0.8396 \pm 0.0058$ | $0.8576 \pm 0.0138$ |
| JDsearch | Src      | $0.8357 \pm 0.0026$ | $0.9627 \pm 0.0069$ | $0.9783 \pm 0.0052$ | $0.9160 \pm 0.0104$ | $0.9284 \pm 0.0068$ |

CFC is a competitive reference point. For recommendation on KuaiSAR, it outperforms 8 of the 11 published baselines on NDCG@5. For search, among the published baselines, only SESRec performs better on all five KuaiSAR metrics and on JDsearch HR@1 and NDCG@5. Conventional composition still does not close the gap to MtlSAR. On KuaiSAR, MtlSAR raises NDCG@5 from 0.1494 to 0.1757 for recommendation and from 0.4152 to 0.5514 for search, corresponding to relative gains of 17.6% and 32.8%. These results show that conventional composition alone does not account for the reported gains.

Distinction from the five lines named by the reviewer:

- UniSAR. Its modeling unit is the fine-grained transition between search and recommendation behavior states, and its PLE experts are static MLPs applied to the already-encoded history. MtlSAR keeps the event fields as the modeling unit, and target conditioning happens before the expert transformation: the candidate item, or the query-item pair in search, attends to the shared event memory first, so different targets read different evidence from the same history.
- UnifiedSSR. It encodes product and query histories in two parallel branches, which preserves query history but not multi-behavior evidence, and it does not serve both workloads with one parameter set (Table II, columns "Multi-behavior" and "One Model").
- SESRec. It is a search-enhanced recommendation model that disentangles similar and dissimilar interests; it treats interactions as binary feedback and does not predict search natively (Table II, column "Task-Src"). In our experiments it can rank search candidates only after we attach a parallel search head.
- Multi-behavior Transformers (MB-STR, NextIP, EIDP). They model behavior types as item-level labels within a single recommendation stream and reduce search to a static behavior ID, so query semantics and the query-click-exposure structure of a search event are not used at all.
- Target-aware MoE / MMoE / PLE. Their gates are conditioned on the input representation, while the experts operate on a target-independent history encoding. In MtlSAR the memory read itself is target-conditioned, which is what the ablation isolates.

Two of the three mechanisms differ from their standard counterparts rather than reusing them:

- BAMHA is not standard masked attention. The mask is behavior-exclusive rather than positional or causal: the $H$ heads are partitioned into intra-behavior groups $\mathcal{H}_{\text{intra}}^b$, inter-behavior heads $\mathcal{H}_{\text{inter}}$ and general heads $\mathcal{H}_{\text{gen}}$, and admissibility is decided by whether $\mathrm{type}(x_i)\cap\mathrm{type}(x_j)$ is empty. Adding a behavior type adds one head group, not a new encoder.
- The event-level alignment is not InfoNCE. $\mathcal{L}_{\text{micro}}$ is an ordinal logit regression with learnable ordered thresholds $\theta_1 < \theta_2$ over three graded relevance levels, using exposed-but-unclicked items as the middle level; InfoNCE is used only at the user level, in $\mathcal{L}_{\text{macro}}$. This is precisely because InfoNCE would push partially relevant items as far away as random ones.

The ablation in Table VI is consistent with this reading. Removing BAMHA reduces KuaiSAR NDCG@5 from 0.1757 to 0.1568 for recommendation and from 0.5514 to 0.4387 for search. Replacing the target-aware experts with a standard PLE layer reduces the two values to 0.1066 and 0.4061. Removing $\mathcal{L}_{\text{micro}}$ reduces them to 0.1638 and 0.4320.

### D2: Efficiency, scalability, and deployment costs are not evaluated. MtlSAR includes multiple stacked attention modules, bidirectional cross-attention, several task-specific and shared experts, input gating, and multiple auxiliary losses. However, the paper provides no comparison of: parameter count, training time, inference latency, GPU memory usage, throughput, complexity with respect to history length and number of behavior types. This omission is important because the paper motivates a unified model partly through reduced duplication and parameter efficiency. The claim that one shared model is operationally advantageous is not supported without an explicit cost analysis.
RD2. We agree this is a necessary check, especially because reduced duplication is one motivation for unification. We compare MtlSAR with UniSAR and SESRec, the two strongest baselines overall. All measurements use one NVIDIA L20 (46 GB) in FP32, a training batch size of 1,024, and an inference batch size of 512. We time inference with CUDA events over 100 batches after 20 warm-up batches, excluding data loading, host-to-device transfer, and metric computation.

| JDsearch metric | MtlSAR | UniSAR | SESRec |
| :--- | ---: | ---: | ---: |
| Parameters | 20.60 M | 17.52 M | 16.85 M |
| Time per epoch | 16.56 min | 11.92 min | 7.74 min |
| Total training time | 3.99 h | 3.11 h | 2.03 h |
| Peak training memory | 6.14 GiB | 5.46 GiB | 3.31 GiB |
| Peak inference memory | 1.69 GiB | 3.03 GiB | 1.18 GiB |
| Latency, rec / src | 0.0961 / 0.0955 ms | 0.1136 / 0.1272 ms | 0.0625 / 0.0678 ms |
| Throughput, rec / src | 10,404 / 10,475 QPS | 8,799 / 7,859 QPS | 16,004 / 14,745 QPS |

| KuaiSAR metric | MtlSAR | UniSAR | SESRec |
| :--- | ---: | ---: | ---: |
| Parameters | 44.91 M | 41.82 M | 41.15 M |
| Time per epoch | 6.91 min | 6.72 min | 5.22 min |
| Total training time | 3.36 h | 2.75 h | 3.42 h |
| Peak training memory | 4.55 GiB | 3.78 GiB | 2.40 GiB |
| Peak inference memory | 1.62 GiB | 2.96 GiB | 1.11 GiB |
| Latency, rec / src | 0.0838 / 0.0859 ms | 0.1107 / 0.1129 ms | 0.0661 / 0.0620 ms |
| Throughput, rec / src | 11,927 / 11,641 QPS | 9,031 / 8,856 QPS | 15,137 / 16,135 QPS |

MtlSAR has between 7.4% and 22.3% more parameters than the baselines but remains in the same order of magnitude. Shared input representations account for 76.3% of the parameters on JDsearch and 89.1% on KuaiSAR. For clarity, the item lookup table stores a 32-dimensional ID embedding that is concatenated with category and shop embeddings and then projected to the $d=128$ event dimension. Training is more expensive per epoch. On JDsearch, the three models converge after a comparable number of epochs, so MtlSAR's longer total time mainly reflects its higher per-epoch cost. At serving time, MtlSAR uses less memory and has lower latency than UniSAR, but it is slower than SESRec, which is lighter and does not serve search natively.

On the duplication argument specifically: the shared input representation amounts to 15.72 M parameters on JDsearch and 40.02 M on KuaiSAR. Deploying two independent single-task models would duplicate this block in full, on top of two separate encoders and two serving paths, whereas MtlSAR serves both workloads from one parameter set. Separate training is also worse in accuracy: the "w/o Joint Training" ablation in Table VI drops KuaiSAR NDCG@5 from 0.1757 to 0.1579 for recommendation and from 0.5514 to 0.3524 for search. MtlSAR therefore avoids duplicating the shared input block and outperforms the separate-training ablation in our setting.

**Scalability.** Let $M$ be the batch size, $A$ the number of attention layers, $L$ the total history length, $|\mathcal{B}|$ the number of behavior types, $N$ the number of candidates, $T$ the number of tasks, and $E_s$ and $E_t$ the numbers of shared and task-specific experts:

- MtlSAR: $O(MA(L^2d+Ld^2) + ML^2|\mathcal{B}| + M(T+1)NLd + M(E_s+TE_t)Nd^2)$
- UniSAR: $O(M[A(L^2d+Ld^2) + L|\mathcal{B}|d + NLd^2])$
- SESRec: $O(M[A(L^2d+Ld^2) + L^2d + L|\mathcal{B}|d + NLd])$

All three models contain an $O(L^2)$ self-attention term and grow linearly with $|\mathcal{B}|$. Two properties of our design matter here. The behavior masks add an $O(L^2|\mathcal{B}|)$ term but no parameters because a new behavior type uses one intra-behavior head group rather than another encoder. Target attention also aggregates the history before the experts, so the expert count does not multiply the $L^2$ term. Under our configuration, $L=60$ (30 events per scenario), $d=128$, $|\mathcal{B}|=4$, $T=2$, and $E_s=E_t=4$, with one attention layer. The candidate-dependent cost per candidate is about $2.2\times10^5$ for MtlSAR and $9.8\times10^5$ for UniSAR. This difference is one of the main reasons for the lower peak inference memory reported above. The auxiliary losses, including the in-batch contrastive term, are used only during training. One scalability limit is that BAMHA requires $H\ge|\mathcal{B}|+2$ heads. With a much larger behavior vocabulary, one would need to increase $d$ or group behaviors into broader types.

### D3: The experimental protocol based on 99 sampled negatives is insufficient for strong ranking claims. The evaluation ranks one positive item against only 99 randomly sampled negatives. This protocol can substantially inflate HR and NDCG values and may not reflect full-catalog retrieval performance. This concern is particularly visible on JDsearch, where several methods achieve HR@10 values near or above 0.98. The paper does not evaluate full-corpus retrieval, hard-negative ranking, approximate nearest-neighbor retrieval, or industrial-scale candidate generation. Consequently, it is unclear whether the observed gains would persist under a realistic ranking workload.
RD3. We accept the substance of this comment: sampled metrics inflate absolute values and compress differences between methods, and the reviewer's observation about JDsearch is correct.

**Scope of the claim.** MtlSAR is a re-ranking model rather than a first-stage retriever. Its score uses target-aware attention and task-specific prediction layers, as shown in Eqs. (13)-(17), and cannot be reduced to a query-item dot product. It is therefore intended for a candidate set produced upstream, not for ANN retrieval or candidate generation. We do not claim full-catalog retrieval performance and will state this scope clearly in the paper.

**Why 99 negatives were used.** The protocol was chosen for comparability. SESRec [4], UnifiedSSR [3], UniSAR [1], and the recent LCR-SER [2] pair the ground-truth item with 99 randomly sampled negatives. USER [5] uses a different protocol: 9 negatives selected by popularity and topic similarity for recommendation, and re-ranking of the real top-20 impression list for search. Among the S&R baselines, only JSR [11] differs. The same 99-negative protocol is also common in sequential recommendation. Using a different candidate construction only for MtlSAR would prevent a fair comparison in Tables IV-V. Full ranking is also costly for our scorer. The shared history encoding can be reused, but the target-dependent readout must be recomputed for every candidate, for every test instance and seed.

**Additional evidence.** We repeated the evaluation with 999 randomly sampled negatives for all methods. Because of space limits, the full table is in the README of our repository. The relative ordering of methods and the conclusions in the paper are unchanged.

**Where our conclusions rest.** The saturated cells are exactly the ones we do not use to support significance claims: on JDsearch, we report no significant improvement at recommendation HR@10 or at search HR@5 and HR@10. The gains concentrate on less saturated metrics. On KuaiSAR, MtlSAR improves search HR@1 by 21.4% and recommendation HR@1 by 13.9% over the second-best method, while the difference on JDsearch search HR@10 is only 0.26%. KuaiSAR shows no saturation, with recommendation HR@1 ranging from 0.054 to 0.079 across all methods. Sampled evaluation therefore compresses our reported gains rather than producing them.

**On hard negatives.** Exposed-but-unclicked items are the natural hard-negative pool in these datasets, but they are also the items supervised by $\mathcal{L}_{\text{micro}}$ in MtlSAR. The baselines do not receive this supervision. Building the evaluation candidate set from these items would therefore favor our model, so we kept a protocol that is neutral with respect to our own training signal.

# References

[1] Teng Shi, Zihua Si, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Dewei Leng, Yanan Niu, and Yang Song. 2024. UniSAR: Modeling User Transition Behaviors between Search and Recommendation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1029-1039.

[2] Teng Shi, Weicong Qin, Weijie Yu, Xiao Zhang, Ming He, Jianping Fan, and Jun Xu. 2025. Bridging Search and Recommendation through Latent Cross Reasoning. arXiv preprint arXiv:2508.04152.

[3] Jiayi Xie, Shang Liu, Gao Cong, and Zhenzhong Chen. 2024. UnifiedSSR: A Unified Framework of Sequential Search and Recommendation. In Proceedings of the ACM Web Conference 2024. 3410-3419.

[4] Zihua Si, Zhongxiang Sun, Xiao Zhang, Jun Xu, Xiaoxue Zang, Yang Song, Kun Gai, and Ji-Rong Wen. 2023. When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1313-1323.

[5] Jing Yao, Zhicheng Dou, Ruobing Xie, Yanxiong Lu, Zhiping Wang, and Ji-Rong Wen. 2021. USER: A Unified Information Search and Recommendation Model Based on Integrated Behavior Sequence. In Proceedings of the 30th ACM International Conference on Information and Knowledge Management. 2373-2382.

[6] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In Proceedings of the 26th International Conference on World Wide Web. 173-182.

[7] Wang-Cheng Kang and Julian McAuley. 2018. Self-Attentive Sequential Recommendation. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 197-206.

[8] Yehuda Koren. 2008. Factorization Meets the Neighborhood: A Multifaceted Collaborative Filtering Model. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 426-434.

[9] Jin Huang, Wayne Xin Zhao, Hongjian Dou, Ji-Rong Wen, and Edward Y. Chang. 2018. Improving Sequential Recommendation with Knowledge-Enhanced Memory Networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval. 505-514.

[10] Kun Zhou, Hui Yu, Wayne Xin Zhao, and Ji-Rong Wen. 2022. Filter-enhanced MLP is All You Need for Sequential Recommendation. In Proceedings of the ACM Web Conference 2022. 2388-2399.

[11] Hamed Zamani and W. Bruce Croft. 2018. Joint Modeling and Optimization of Search and Recommendation. In Proceedings of the First Biennial Conference on Design of Experimental Search & Information Retrieval Systems (CEUR Workshop Proceedings, Vol. 2167). 36-41.

[12] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI). 452-461.

[13] Fernando Benjamín Pérez Maurera, Maurizio Ferrari Dacrema, Pablo Castells, and Paolo Cremonesi. 2025. Impression-Aware Recommender Systems. ACM Transactions on Recommender Systems 3, 4 (2025), 1-46.

[14] Jingtao Ding, Guanghui Yu, Xiangnan He, Fuli Feng, Yong Li, and Depeng Jin. 2021. Sampler Design for Bayesian Personalized Ranking by Leveraging View Data. IEEE Transactions on Knowledge and Data Engineering 33, 2 (2021), 667-681.

[15] Ruobing Xie, Cheng Ling, Yalong Wang, Rui Wang, Feng Xia, and Leyu Lin. 2020. Deep Feedback Network for Recommendation. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI). 2519-2525.

[16] Thorsten Joachims, Adith Swaminathan, and Tobias Schnabel. 2017. Unbiased Learning-to-Rank with Biased Feedback. In Proceedings of the Tenth ACM International Conference on Web Search and Data Mining. 781-789.

[17] Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata. 2020. Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback. In Proceedings of the 13th International Conference on Web Search and Data Mining. 501-509.

[18] Wenjie Wang, Fuli Feng, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. 2021. Denoising Implicit Feedback for Recommendation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining. 373-381.

[19] Thorsten Joachims, Laura Granka, Bing Pan, Helene Hembrooke, and Geri Gay. 2017. Accurately Interpreting Clickthrough Data as Implicit Feedback. ACM SIGIR Forum 51, 1 (2017), 4-11.
