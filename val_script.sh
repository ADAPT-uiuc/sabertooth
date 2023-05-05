#! /bin/bash

## This will validate all the checkpoints.
for ((i=0; i<24; i+=1))
do
        dir="/home/ag82/sabertooth/output/pretrain_20230504_1358/${i}"
        cd $dir
        for file in $(ls)
        do
            python run_pretraining.py --config=configs/pretraining.py --config.init_checkpoint="$dir/$file" --config.do_train=False --config.do_eval=True
            python collator.py "$dir"
        done
done