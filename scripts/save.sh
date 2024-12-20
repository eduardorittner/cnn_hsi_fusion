python train.py --save --model mst --save-dir results/mst/ --pretrained-path exp/mstexp1/net_30epoch.pth
wait
python train.py --save --model mst_plus_plus --save-dir results/mst-plus/ --pretrained-path exp/mst-plus-exp1/net_47epoch.pth
wait
python train.py --save --model mprnet --save-dir results/mprnet/ --pretrained-path exp/mprnet-exp2/net_149epoch.pth
wait
python train.py --save --model mirnet --save-dir results/mirnet/ --pretrained-path exp/mirnet-exp2/net_42epoch.pth
wait
python train.py --save --model hdnet --save-dir results/hdnet/ --pretrained-path exp/hdnet-exp1/net_39epoch.pth
wait
python train.py --save --model restormer --save-dir results/restormer/ --pretrained-path exp/restormer-exp3/net_150epoch.pth
wait
