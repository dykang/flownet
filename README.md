# FlowNet
Data and code for ["Linguistic Versus Latent Relations for Modeling a Flow in Paragraphs"](https://arxiv.org/) by Dongyeop Kang, Hiroaki Hayashi, Alan W Black, and Eduard Hovy, EMNLP 2019

## Dependency
 - tensorflow
 - nltk
 - gensim
 - nlgeval
 
## Notes
 - (Aug 2019) This repository is not heavily tested due to the major refactoring of flownet with pytorch and transfomers. Please stay tuned for the new pytorch version of flownet. 


## Setup Configuration
Run `./setup.sh` at the root of this repository to install dependencies.

## Dataset preparation
XXX

## Models
In order to experiment with (and hopefully improve) our models, you can run following commands:

To run flownet with delta relations, run 
```
    python train_lm.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET_NAME} \
        --model_path ${MODEL_PATH} \
        --model_name drnn
```

To run flownet with discourse relations, run 
```
    python train_lm.py \
        --data_path ${DATA_PATH} \
        --dataset ${DATASET_NAME} \
        --model_path ${MODEL_PATH} \
        --model_name rstseq2seq
```

You can add ```--reload``` or ```--test``` options to reload the pre-trained model or test on testing data, respectively. 


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

