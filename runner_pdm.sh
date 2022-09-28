#!/bin/bash

#python -m src.load_pdm --nameParams $1

echo "--------------------------------- finish load pdm ---------------------------------"


#python -m src.extract_pdm --nameParams $1 --extractParams $2 --nbCores $4


echo "--------------------------------- finish extract pdm ---------------------------------"

#python -m src.train_pdm --nameParams $1 --extractParams $2 --nbCores $4

echo "--------------------------------- finish train pdm ---------------------------------"


#python -m models.eco_h.$3.params


python -m src.train_economy_h_pdm --exp $3 --nbCores $4
echo "--------------------------------- finish train eco ---------------------------------"

python -m src.baselines_competitors_pdm --nameParams $1 --extractParams $2 --expeco $3 --nbCores $4 
echo "---------------------------------



