#!/bin/bash
echo "start executing"
eval "$(conda shell.bash hook)"
conda activate dynamic_entropy
for notebook in "$@"; do
    echo "execute: $notebook"
    if jupyter nbconvert --execute --inplace "$notebook"; then
        echo "success: $notebook"
    else
        echo "failure: $notebook"
    fi 
done
echo "Done"