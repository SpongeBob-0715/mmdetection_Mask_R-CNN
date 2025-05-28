
# 创建日志目录（如果不存在）
mkdir -p ../work_dirs/log/

# 启动训练任务
CUDA_VISIBLE_DEVICES=7 nohup python tools/train.py ../sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py > ../work_dirs/log/train.log 2>&1 &

# 提示信息
echo "训练已启动，日志文件：../work_dirs/log/train.log"
echo "查看实时日志：tail -f ../work_dirs/log/train.log"