python ../train.py --method mst --end-epoch 250 --outf ../exp --data-root ../data/ --patch-size 128 --stride 64
python ../train.py --method mst_plus_plus --end-epoch 250 --outf ../exp --data-root ../data/ --patch-size 128 --stride 64
python ../train.py --method restormer --end-epoch 250 --outf ../exp --data-root ../data/ --patch-size 128 --stride 64
python ../train.py --method hdnet --end-epoch 250 --outf ../exp --data-root ../data/ --patch-size 128 --stride 64
