#!/usr/bin/env bash
wget http://www-mmsp.ece.mcgill.ca/Documents/Data/TSP-Speech-Database/8k-G712.zip
tar -xvzf TEDLIUM_release1.tar.gz
mkdir -p ~/vitaflow/TEDLiumDataset/raw_data/
cp -r TEDLIUM_release1/train/ ~/vitaflow/TEDLiumDataset/raw_data/
cp -r TEDLIUM_release1/dev/ ~/vitaflow/TEDLiumDataset/raw_data/
cp -r TEDLIUM_release1/test/ ~/vitaflow/TEDLiumDataset/raw_data/

