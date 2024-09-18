# Machine Learning for Quantum Error Correction

In this repo, we developed BERT-QEC, a novel application of bidirectional transformer models for quantum error correction. Building on the principles of masked language modeling (MLM) introduced in BERT, we propose a new masking scheme designed to train a transformer encoder that treats the decoding step of quantum error correction as a classification task—specifically, determining whether a qubit error is correctable. Unlike previous approaches that train separate models for different quantum code distances, BERT-QEC leverages a single model trained across multiple code distances simultaneously. This not only simplifies the training process but also improves scalability. We further investigate the generalization capabilities of our model on unseen code distances and evaluate the impact of fine-tuning on these novel distances. Our results demonstrate that BERT-QEC offers a promising step toward efficient and adaptable quantum error correction, with significant potential for real-world quantum computing applications.

Warm-up examples based on the following papers:

https://arxiv.org/abs/2202.05741

https://arxiv.org/abs/2311.16082

Usage
------------------

You need pytorch and GPU.

.py files are source codes.

gen_data.py generates training, validation, and test datasets. Make sure to uncomment the relevant blocks.
