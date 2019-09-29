#!/usr/bin/env bash
wget https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release1.tar.gz
tar -xvzf TEDLIUM_release1.tar.gz
mkdir -p ~/vitaflow/TEDLiumDataset/raw_data/
cp -r TEDLIUM_release1/train/ ~/vitaflow/TEDLiumDataset/raw_data/
cp -r TEDLIUM_release1/dev/ ~/vitaflow/TEDLiumDataset/raw_data/
cp -r TEDLIUM_release1/test/ ~/vitaflow/TEDLiumDataset/raw_data/

