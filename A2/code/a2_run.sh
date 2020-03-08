#!/bin/bash
# Setup: chmod +x a2_run.sh

# For a full run, run `./a2_run.sh cs init all`

# For initialization only, run `./a2_run.sh cs init`
# For a debug run, run `./a2_run.sh cs skip wo` (or `w`, `all`)

if [ "$1" == "local" ]; then
    echo '--- Running locally ---'
    TRAIN=/Users/joanna.zyz/NLP-Applications/A2/data/Hansard/Training/
    TEST=/Users/joanna.zyz/NLP-Applications/A2/data/Hansard/Testing/
else # "cs"
    echo '--- Running on CS ---'
    TRAIN=/h/u1/cs401/A2/data/Hansard/Training/
    TEST=/h/u1/cs401/A2/data/Hansard/Testing/
fi

if [ "$2" == "init" ]; then
    echo '--- Preparation ---'
    # 1.
    python3.7 a2_run.py vocab $TRAIN e vocab.e.gz
    python3.7 a2_run.py vocab $TRAIN f vocab.f.gz
    # 2.
    python3.7 a2_run.py split $TRAIN train.txt.gz dev.txt.gz
else # "skip"
    echo '--- Preparation skipped ---'
fi


if [ "$3" == "wo" ]; then
    echo '--- Without attention ---'
    # 3.
    python3.7 a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_wo_att.pt.gz --device cuda
    # 5.
    python3.7 a2_run.py test $TEST vocab.e.gz vocab.f.gz model_wo_att.pt.gz --device cuda

elif [ "$2" == "w" ]; then
    echo '--- With attention ---'
    # 4.
    python3.7 a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_w_att.pt.gz --with-attention --device cuda
    # 6.
    python3.7 a2_run.py test $TEST vocab.e.gz vocab.f.gz model_w_att.pt.gz --with-attention --device cuda


elif [ "$2" == "all" ]; then
    echo '--- Both with and without attention ---'
    # 3.
    python3.7 a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_wo_att.pt.gz --device cuda
    # 4.
    python3.7 a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_w_att.pt.gz --with-attention --device cuda
    # 5.
    python3.7 a2_run.py test $TEST vocab.e.gz vocab.f.gz model_wo_att.pt.gz --device cuda
    # 6.
    python3.7 a2_run.py test $TEST vocab.e.gz vocab.f.gz model_w_att.pt.gz --with-attention --device cuda

else
    echo '--- No train/test ---'
fi


# # 3.
# python3.7 a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_wo_att.pt.gz --device cuda

# # 4.
# python3.7 a2_run.py train $TRAIN vocab.e.gz vocab.f.gz train.txt.gz dev.txt.gz model_w_att.pt.gz --with-attention --device cuda

# # 5.
# python3.7 a2_run.py test $TEST vocab.e.gz vocab.f.gz model_wo_att.pt.gz --device cuda

# # 6.
# python3.7 a2_run.py test $TEST vocab.e.gz vocab.f.gz model_w_att.pt.gz --with-attention --device cuda

# Code from cs server: /h/u1/cs401/A2/code
# Copy from server to local: 
#   Folder: scp -r zhouyiz2@teach.cs.utoronto.ca:/h/u1/cs401/A2/code Downloads/
#   File: scp zhouyiz2@teach.cs.utoronto.ca:/h/u1/cs401/A2/code/a2_abcs.py Downloads/
