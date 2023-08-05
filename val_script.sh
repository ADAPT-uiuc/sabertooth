#! /bin/bash

## This will validate all the checkpoints.
for ((i=0; i<11; i+=1))
do
	dir="/home/ag82/sabertooth/output/pretrain_20230513_1445/${i}"
        cd $dir
        for file in $(ls)
        do
            python /home/ag82/sabertooth/run_pretraining.py --config=/home/ag82/sabertooth/configs/pretraining.py --config.init_checkpoint="$dir/$file" --config.do_train=False --config.do_eval=True --output_dir $dir
            python /home/ag82/sabertooth/collator.py "$dir"
        done
done


## For testing purposes only. 

#python /home/ag82/sabertooth/run_pretraining.py --config=/home/ag82/sabertooth/configs/pretraining.py --config.init_checkpoint="/home/ag82/sabertooth/output/pretrain_20230504_1358/0/checkpoint_10579" --config.do_train=False --config.do_eval=True --output_dir "/home/ag82/sabertooth/output/pretrain_20230504_1358/0"
