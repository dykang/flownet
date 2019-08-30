#!/bin/bash

# fotmat:
# topic \t fid \t sent_len \t avg_len \t [sent1 \t sent2 \t sent3 ...] \n

OUT_FILE="/data/paragen/data/"

MAX_SENT=7
MIN_SENT=4
MAX_SENT_LEN=40
MIN_SENT_LEN=5

python prepare_book.py \
  -data_dir /data/lm/BookCorpus/ \
  -out_file ${OUT_FILE}book \
  -max_sent ${MAX_SENT} \
  -min_sent ${MIN_SENT} \

python prepare_arxiv.py \
  -data_dir /data/AutoReview \
  -out_file ${OUT_FILE}arxiv \
  -max_sent ${MAX_SENT} \
  -min_sent ${MIN_SENT} \
  -max_sent_len ${MAX_SENT_LEN} \
  -min_sent_len ${MIN_SENT_LEN}


