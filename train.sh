model_name=1
epoch=2
eval=1
embd=128

python main.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model NPCT 
python main.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model SPCT
python main.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT
python main.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT_FP
python main.py --epoch $epoch --eval $eval --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT_FPADV