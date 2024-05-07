model_name=6
epoch=400
eval=1
embd=64

# python train.py --epoch 100 --eval $eval --lr 0.0005 --dp 0.6 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FPMOD 
# python train.py --epoch 100 --eval $eval --lr 0.0002 --dp 0.5 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FPMOD --load_checkpoint True
# python train.py --epoch 100 --eval $eval --lr 0.0001 --dp 0.4 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FPMOD --load_checkpoint True
# python train.py --epoch 100 --eval $eval --lr 0.00009 --dp 0.3 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FPMOD --load_checkpoint True
# python train.py --epoch 100 --eval $eval --lr 0.00009 --dp 0.2 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FPMOD --load_checkpoint True
# python train.py --epoch 100 --eval $eval --lr 0.00009 --dp 0.1 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FPMOD --load_checkpoint True

python train.py --epoch 100 --eval $eval --lr 0.0005 --dp 0.6 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FP 
python train.py --epoch 100 --eval $eval --lr 0.0002 --dp 0.5 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FP --load_checkpoint True
python train.py --epoch 100 --eval $eval --lr 0.0001 --dp 0.4 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FP --load_checkpoint True
python train.py --epoch 100 --eval $eval --lr 0.00009 --dp 0.3 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FP --load_checkpoint True
python train.py --epoch 100 --eval $eval --lr 0.00009 --dp 0.2 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FP --load_checkpoint True
python train.py --epoch 100 --eval $eval --lr 0.00009 --dp 0.1 --step_size 25 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model PCT_FP --load_checkpoint True

# python train.py --epoch $epoch --eval $eval --lr 0.001 --step_size 20 --points_taken 4096 --batch_size 32 --grid_size 25 --model_name $model_name --model NPCT 
# python train.py --epoch $epoch --eval $eval --lr 0.001 --step_size 20 --points_taken 4096 --batch_size 16 --grid_size 25 --model_name $model_name --model SPCT
# python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT
# python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 10 --points_taken 4096 --batch_size 32 --grid_size 25 --model_name $model_name --model PCT_FP
