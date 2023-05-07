#!/bin/bash

python scripts/train_pos_tagging.py -train_file "$1" -dev_file "$2" -model_checkpoint "$3" -output_dir "$4"