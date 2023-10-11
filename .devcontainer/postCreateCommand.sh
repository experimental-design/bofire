#!/bin/bash

conda init
conda install -f conda.yml
conda activate base
# conda activate bofire
pip install -r requirements.txt