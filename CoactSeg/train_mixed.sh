nvidia-smi


python ./code/train_3d_mix.py --gpu 0 --max_iteration 250 --loss_func "hyTver" --act "leakyRelu"

python ./code/test_3d_mix.py --gpu 0 --loss_func "hyTver" --act "leakyRelu" --iter_num 250
