## train
### mask his
python main.py --dataname ML20M --batch_size 500 --topk 10 --mask_his --device 1 
python main.py --dataname Netflix --batch_size 500 --topk 10 --mask_his --device 2
python main.py --dataname MSD --batch_size 500 --topk 10 --mask_his --device 2

### no mask
python main.py --dataname Koubei  --batch_size 500 --start_from_1 --device 4
python main.py --dataname Tmall  --batch_size 500  --start_from_1 --device 1 



## test
### mask his
python test.py --dataname ML20M --batch_size 500 --mask_his --topk 50 --load_dir "./runs/2021-11-06T17-03-49" --device 1 
python test.py --dataname Netflix --batch_size 500  --mask_his --topk 50 --load_dir ./runs/2021-11-06T17-05-40 --device 1
python test.py --dataname MSD --batch_size 500  --mask_his --topk 50 --load_dir ./runs/2021-11-06T17-08-00  --device 2

### no mask
python test.py --dataname Koubei  --batch_size 500 --start_from_1 --topk 50 --load_dir ./runs/2021-11-06T18-18-12 --device 1
python test.py --dataname Tmall  --batch_size 500  --start_from_1 --topk 50 --load_dir ./runs/2021-11-06T18-19-19 --device 1 
