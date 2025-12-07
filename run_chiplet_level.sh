#!/usr/bin/env bash

chiplet_id=$1
./booksim Chiplet_level_config > out_chiplet_${chiplet_id}.txt
#echo "BookSim output saved for NoC to out_chiplet_${chiplet_id}.txt"
echo -e "\033[1A\r"