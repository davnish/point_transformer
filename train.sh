model_name=1
epoch=10
eval=1
embd=64

# python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model NPCT 
# python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model SPCT
# python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT
python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 10 --points_taken 4096 --batch_size 32 --grid_size 25 --model_name $model_name --model PCT_FP
python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 10 --points_taken 4096 --batch_size 32 --grid_size 25 --model_name $model_name --model PCT_FPMOD
# python train.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT_FPADV