CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node 4 \
train.py -m ./model_config.json -t ./config_sft.json
