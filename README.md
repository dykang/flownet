# FlowNet
Data and code for ["Linguistic Versus Latent Relations for Modeling a Flow in Paragraphs"](https://arxiv.org/) by Dongyeop Kang, Hiroaki Hayashi, Alan W Black, and Eduard Hovy, EMNLP 2019

## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies.

## Dataset preparation
XXX

## Models
In order to experiment with (and hopefully improve) our models, you can run following commands:

To run flownet with delta (--model_name seq2seq) or discourse (--model_name rstseq2seq) relations, run 
```
    python train_clm.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET_NAME} \
        --model_path ${MODEL_PATH} \
        --model_name {seq2seq,rtseq2seq}
```

To run other baseline models (--model_name {rnn,hrnn,hred}), run 
```
    python train_clm.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET_NAME} \
        --model_path ${MODEL_PATH} \
        --model_name {rnn,hrnn,hred}
```

## Citation
    
    @inproceedings{kang19emnlp_flownet,
      title = {Linguistic Versus Latent Relations for Modeling a Flow in Paragraphs},
      author = {Dongyeop Kang and Hiroaki Hayashi and Alan W Black and Eduard Hovy},
      booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
      address = {Hong Kong},
      month = {November},
      url = {https://arxiv.org/},
      year = {2019}
    }

## Acknowledgement
 - We use some of the code in XXX for XXX.
 - We also thank Jason Weston, Dan Jurafsky, and anonymous reviewers for their helpful comments.

