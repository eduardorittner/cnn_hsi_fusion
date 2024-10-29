python ../train.py --model restormer --end-epoch 150 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64 --batch 10
wait
python ../train.py --model mprnet --end-epoch 150 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64 --batch 40
wait
python ../train.py --model mirnet --end-epoch 150 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64 --batch 40
wait
python ../train.py --model hinet --end-epoch 150 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64 --batch 40
wait
