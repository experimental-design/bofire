#!/bin/bash

conda env create -f conda.yml
conda activate bofire
pip install -r requirements.txt