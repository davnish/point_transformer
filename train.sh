model_name = 1
epoch = 2

python main.py --epoch $epoch --eval 1 --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model NPCT 
python main.py --epoch $epoch --eval 1 --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model SPCT
python main.py --epoch $epoch --eval 1 --lr 0.0001 --step_size 100 --points_taken 4096 --batch_size 8 --grid_size 25 --model_name $model_name --model PCT