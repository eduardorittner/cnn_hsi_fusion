python ../train.py --model mst --end-epoch 200 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64
wait
python ../train.py --model mst_plus_plus --end-epoch 200 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64
wait
python ../train.py --model restormer --end-epoch 200 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64
wait
python ../train.py --model hdnet --end-epoch 200 --outf ../exp/ --data-root ../data/ --patch-size 128 --stride 64
wait
