# adapter-subgroup-analysis

This repository contains the code for the paper

> D. Gupta, E. A. M. Stanley, F. De Sousa Ribeiro, S. R. Desai, B. Glocker
> **[Subgroup performance analysis of adaptation strategies for chest X-ray foundation models](https://arxiv.org/abs/2608.19078)**
> MICCAI 2026 FAIMI Workshop.


We compare three different adaptation methods on top of a frozen ViT-based foundation model and see how both overall and subgroup performance across clinical tasks are affected. We investigate **No Adapter** (a linear head on the raw CLS token), **MLP** (a hidden-layer projection of the CLS token), and **Attention Pooling** (a learned query attending over CLS + patch tokens
concatenated across four transformer layers). Additiaonlly, we look at how the layer selection for **Attention Pooling** impacts subgroup disparities.

## Code

This repo contains the training, evaluation and figure-generation code.

```
adapter-subgroup-analysis/
├── paths.py                # <- edit this to point at your own data/output locations
├── training/                # model definitions, dataset loading, training loops
├── evaluation/               # shared eval library + attention-pooling eval scripts
├── scripts/                  # thin shell launchers, one per pipeline stage
└── notebooks/
    └── figures.ipynb         # reproduces Table 1, Figure 2, Table 2 and Figure 3
```

### Setup

Create and activate a Python 3.10+ environment, then install dependencies:

```shell
conda create -n adapter-subgroup-analysis python=3.10
conda activate adapter-subgroup-analysis
pip install -r requirements.txt
```

### Data

The MIMIC-CXR imaging dataset can be downloaded from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ with the corresponding demographic information available from https://physionet.org/content/mimiciv/1.0/.



### Configure paths

Edit [`paths.py`](paths.py) to point at your image directory, metadata CSV, and
where you want checkpoints/embeddings/outputs written — or set the matching
`ADAPTER_*` environment variables. Every script in `training/`, `evaluation/` and
`scripts/` reads its defaults from this one file.

### Reproduce the paper

1. **Extract CLS embeddings** (required before step 2 and the notebook's inline
   scoring cells — Attention Pooling always uses raw images and never needs this):
   ```shell
   bash scripts/extract_cls_embeddings.sh
   ```
2. **Train No Adapter and MLP** — 8 pathology tasks only, directly on the cached
   embeddings. Edit the toggle in the script and run twice:
   ```shell
   bash scripts/train_adapter.sh   # ADAPTER_TYPE=none, then ADAPTER_TYPE=mlp
   ```
   Then fit their post-hoc attribute (race/sex/view) probes, again twice:
   ```shell
   bash scripts/fit_attribute_probes.sh
   ```
3. **Train the 4 Attention Pooling layer configs** — Early (2,3,4,5), Late
   (9,10,11,12), Split (2,3,11,12), Even (3,6,9,12). Edit the toggle and run 4 times:
   ```shell
   bash scripts/train_attn.sh
   ```
   Then train each config's frozen attribute probe, again 4 times:
   ```shell
   bash scripts/train_attn_attribute_probes.sh
   ```
4. **Evaluate the Attention Pooling checkpoints** (No Adapter/MLP are scored
   inline in the notebook instead — see step 6):
   ```shell
   bash scripts/eval_attn_multilayer.sh
   bash scripts/eval_attn_attributes.sh
   ```
5. **Compute 95% bootstrap confidence intervals** for the two summary tables
   (100 resampling iterations of the frozen test set):
   ```shell
   bash scripts/eval_bootstrap_ci.sh
   ```
6. **Open the notebook** and run all cells:
   ```shell
   jupyter notebook notebooks/figures.ipynb
   ```
   This produces, in order: a dataset prevalence plot, **Table 1** (overall
   pathology + attribute AUROC with 95% CIs, No Adapter vs MLP vs Attention
   Pooling) with a companion bar chart, **Figure 2** (subgroup disparity heatmap
   for the same 3 methods), **Table 2** (the same table across the 4 layer
   configs) with a companion bar chart, and **Figure 3** (its disparity heatmap).



## Citation

```bibtex
@inproceedings{gupta2026subgroup,
  title     = {Subgroup performance analysis of adaptation strategies for chest X-ray foundation models},
  author    = {Gupta, Dhruv and Stanley, Emma A. M. and De Sousa Ribeiro, Fabio and Desai, Sujal R. and Glocker, Ben},
  booktitle = {MICCAI 2026 FAIMI Workshop},
  year      = {2026}
}
```

## Funding

This work was supported by the Royal Academy of Engineering as part of the
Kheiron/RAEng Research Chair. D.G. is supported by UKRI AI Centre for Doctoral
Training in Digital Healthcare [EP/Y030974/1]. E.A.M.S. acknowledges funding from
the Natural Sciences and Engineering Research Council of Canada (NSERC)
Postdoctoral Research Award. F.R. and B.G. acknowledge the support of the UKRI AI
programme and the EPSRC, for CHAI-EPSRC Causality in Healthcare AI Hub
[EP/Y028856/1].

## License

This project is licensed under the [Apache License 2.0](LICENSE).
