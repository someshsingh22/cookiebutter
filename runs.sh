DEVICE=0 nohup python verbalize.py > out0.log &
DEVICE=1 nohup python verbalize.py > out1.log &
DEVICE=2 nohup python verbalize.py > out2.log &
DEVICE=3 nohup python verbalize.py > out3.log &
DEVICE=4 nohup python verbalize.py > out4.log &
DEVICE=5 nohup python verbalize.py > out5.log &
DEVICE=6 nohup python verbalize.py > out6.log &
DEVICE=7 nohup python verbalize.py > out7.log &

nohup python extractscenes.py --num_process 32 --process_idx 0 > out0.log &
nohup python extractscenes.py --num_process 32 --process_idx 1 > out1.log &
nohup python extractscenes.py --num_process 32 --process_idx 2 > out2.log &
nohup python extractscenes.py --num_process 32 --process_idx 3 > out3.log &
nohup python extractscenes.py --num_process 32 --process_idx 4 > out4.log &
nohup python extractscenes.py --num_process 32 --process_idx 5 > out5.log &
nohup python extractscenes.py --num_process 32 --process_idx 6 > out6.log &
nohup python extractscenes.py --num_process 32 --process_idx 7 > out7.log &
nohup python extractscenes.py --num_process 32 --process_idx 8 > out8.log &
nohup python extractscenes.py --num_process 32 --process_idx 9 > out9.log &
nohup python extractscenes.py --num_process 32 --process_idx 10 > out10.log &
nohup python extractscenes.py --num_process 32 --process_idx 11 > out11.log &
nohup python extractscenes.py --num_process 32 --process_idx 12 > out12.log &
nohup python extractscenes.py --num_process 32 --process_idx 13 > out13.log &
nohup python extractscenes.py --num_process 32 --process_idx 14 > out14.log &
nohup python extractscenes.py --num_process 32 --process_idx 15 > out15.log &