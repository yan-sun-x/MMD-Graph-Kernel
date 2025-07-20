#!bin/bash
# This script is used to run the MMD Graph Kernel training and evaluation process.
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee output_vanilla.log
echo "Running Vannila MMD Graph Kernel on MUTAG dataset" | tee -a output_vanilla.log
python -u main.py --model 'vanilla' --dataname 'MUTAG' --dis_gamma 1e0 --bandwidth "[1e0, 1e1]" --gcn_num_layer 2 | tee -a output_vanilla.log
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a output_vanilla.log

echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee output_deep.log
echo "Running Deep MMD Graph Kernel on MUTAG dataset" | tee -a output_deep.log 
python -u main.py --model 'deep' --dataname 'MUTAG' --epochs 10 | tee -a output_deep.log
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a output_deep.log