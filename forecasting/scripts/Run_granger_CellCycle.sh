#!/bin/bash
dataset_name=CellCycle.csv

python3 -u ./Granger_Causality.py --dataset $dataset_name > granger_logs/$dataset_name.log