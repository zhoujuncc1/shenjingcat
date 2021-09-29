#!/usr/bin/env bash
APP="fashion_mlp" #mnist_mlp
MODEL="fashion_2layer.pt"
echo ${APP}.py
echo $MODEL
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 32 --device cuda:0 > ${APP}_log_32bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 8 --device cuda:0 > ${APP}_log_8bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 7 --device cuda:0 > ${APP}_log_7bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 6 --device cuda:0 > ${APP}_log_6bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 5 --device cuda:1 > ${APP}_log_5bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 4 --device cuda:1 > ${APP}_log_4bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 3 --device cuda:1 > ${APP}_log_3bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 2 --device cuda:1 > ${APP}_log_2bit.txt &
# python ${APP}.py --epochs 0 --resume $MODEL --weight-quantize 1 --device cuda:1 > ${APP}_log_1bit.txt &

python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_8bit.pt --weight-quantize 8 --device cuda:0 > ${APP}_log_train_8bit.txt &
python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_7bit.pt --weight-quantize 7 --device cuda:0 > ${APP}_log_train_7bit.txt &
python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_6bit.pt --weight-quantize 6 --device cuda:0 > ${APP}_log_train_6bit.txt &
python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_5bit.pt --weight-quantize 5 --device cuda:1 > ${APP}_log_train_5bit.txt &
python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_4bit.pt --weight-quantize 4 --device cuda:1 > ${APP}_log_train_4bit.txt &
python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_3bit.pt --weight-quantize 3 --device cuda:1 > ${APP}_log_train_3bit.txt &
python ${APP}.py --epochs 5 --resume $MODEL --save-model 2layer_2bit.pt --weight-quantize 2 --device cuda:1 > ${APP}_log_train_2bit.txt &