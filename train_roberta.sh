#!/bin/bash

python scripts/train_roberta.py -train_file "$1" -dev_file "$2" -language "$3"
