# set -x

#################### 网络配置 ###########################
#修改为对应主节点IP
# export MASTER_ADDR=141.61.29.128
# export MASTER_ADDR=141.61.29.129

# 修改为当前节点的通信网卡
SOCKET_IFNAME="data0.3001"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_TIMEOUT=7200000

#################### Log 目录配置 ###########################
# * 确保 JOB_LOG_DIR 在共享盘下
CURRENT_IP=$(ifconfig $TP_SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
# export MASTER_ADDR=$CURRENT_IP  # 单机
export MASTER_ADDR=172.27.1.107  # 多机

export JOB_LOG_DIR=/data/logs/deepseek_671B
export JOB_LOG_DIR_CURR=${JOB_LOG_DIR}/$(date +"%Y%m%d_%H%M%S")
export ASCEND_PROCESS_LOG_PATH=${JOB_LOG_DIR_CURR}/plog/${CURRENT_IP}

####################   环境设置    #######################
DEFAULT_SH=$(realpath $(dirname $0)/run.sh) # 使用与当前sh同目录的run.sh文件
echo $DEFAULT_SH

# python：非交互式终端无法source成功，bug：可以通过临时设置PS1来跳过.bashrc里面的测试
export PS1=111
source /root/.bashrc
unset PS1
conda activate verl_pt27_25rc3

site_packages_dir=$(python -c "import site; print(site.getsitepackages()[0])") # 方便后面对容器里面的代码进行替换
echo $site_packages_dir

# Cann环境
# source /home/cann/8.2.RC2.B030/ascend-toolkit/set_env.sh;
# source /home/cann/8.2.RC2.B030/nnal/atb/set_env.sh;
source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh

# 关闭现有任务
pkill -9 python
ray stop --force
ps -ef | grep "python"| grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {}
pkill -9 torchrun
ps -ef | grep "defunct"|grep python| awk '{print $3}'|xargs -t -i kill -9 {}
ps -ef | grep "defunct"|grep torchrun| awk '{print $3}'|xargs -t -i kill -9 {}
ps -ef | grep -i python |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef | grep -i torchrun |grep -i [name] |grep -v grep |awk '{print $2}' |xargs -t -I {} kill -9 {}
ps -ef | grep "python" | grep -v grep | awk '{print $2}' | xargs -t -i kill -9 {}

rm -rf /tmp/ray/*
export RAY_memory_monitor_refresh_ms=0 # 禁用ray内存监控器、禁用ray任务杀死

# 机器环境变量
export NNODES=1
export NPUS_PER_NODE=16
export GPUS_PER_NODES=$NPUS_PER_NODE

# Debug相关的环境变量
ulimit -n 32768
export RAY_DEBUG=1  # 允许ray debug
export RAY_DEBUG_POST_MORTEM=1
export RAY_DEDUP_LOGS=1  # Ray 日志去重
export HYDRA_FULL_ERROR=1
export ASCEND_GLOBAL_LOG_LEVEL=3  # 3：error级？0：debug级？
export ASCEND_LAUNCH_BLOCKING=0

#! 注意，0929加了这 1 个优化参数， libjemalloc 需要重新编译
# export LD_PRELOAD="/usr/local/lib/libjemalloc.so.2"
export TASK_QUEUE_ENABLE=1

#! 注意，HCCL 相关配置
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

# 确定性计算
# export LCCL_DETERMINISTIC=1
# export HCCL_DETERMINISTIC=true
# export CLOSE_MATMUL_K_SHIFT=1
# export ATB_MATMUL_SHUFFLE_K_ENABLE=0
# export ATB_LLM_LCOC_ENABLE=0

#! #################  【安装新的package】  #####################
# IP=141.5.145.201
# export http_proxy="http://p_atlas:proxy%40123@$IP:8080"
# export https_proxy="http://p_atlas:proxy%40123@$IP:8080"
# export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

# tensordict>=0.10(包含了npu的bug修复) & pystack & mindstudio-probe (精度问题debug) & mbridge (权重加载)
# pip install -U --no-index --find-link=/data/logs/pkgs/tensordict tensordict
# pip install -U --no-index --find-link=/data/logs/pkgs/pystack pystack
# pip install --no-index --find-link=/data/logs/pkgs/mindstudio mindstudio-probe
# pip install --no-index --find-link=/data/logs/pkgs/mbridge mbridge
# pip install --no-index --find-link=/data/logs/pkgs/debugpy debugpy # ray debug

# mbridge相关修复已提，修复好之后可直接使用main分支：https://github.com/ISEEKYAN/mbridge/pull/52，https://github.com/ISEEKYAN/mbridge/pull/53
# rm -rf $site_packages_dir/mbridge
# \cp -r /home/projects/mbridge/mbridge $site_packages_dir/mbridge

export VLLM_VERSION=0.11.0
export VLLM_ASCEND_ENABLE_NZ=0 #影响精度

#! #################  【VLLM】  #####################
#! 规避模型加载时 权重读取错误的问题 vllm 0.11 不适用？
# bash /home/code/verl-gpu/k8s/patch/apply_vllm-ascend.sh
# pip uninstall vllm -y
# cd /home/projects/vllm
# _PYPROJECT_HOOKS_BACKEND_PATH=$site_packages_dir pip install . --no-deps

#! #################  【Megatron】  #####################
#! [Megatron]
# Yarn：use_cpu_initialization
rm -rf $site_packages_dir/megatron
cp -r /home/projects/Megatron-LM/megatron $site_packages_dir/megatron
\cp $(dirname $0)/../patch/megatron.patch/yarn_rotary_pos_embedding.py $site_packages_dir/megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py
echo -e "\033[32m Applied megatron Yarn use_cpu_initialization done. \033[0m"

#! #################  【MindSpeed】  #####################
#! MLA 需要使用新的 mindspeed/2.2.0_core_r0.12.1 消除其原本的bug
rm -rf $site_packages_dir/mindspeed
cp -r /home/projects/MindSpeed/mindspeed $site_packages_dir/mindspeed
echo -e "\033[32m Replace Mindspeed with mindspeed/2.2.0_core_r0.12.1 done. \033[0m"

\cp $(dirname $0)/../patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/te/pytorch/module/grouped_linear.py $site_packages_dir/mindspeed/te/pytorch/module/grouped_linear.py
\cp $(dirname $0)/../patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/core/context_parallel/dot_product_attention.py $site_packages_dir/mindspeed/core/context_parallel/dot_product_attention.py
\cp $(dirname $0)/../patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/core/transformer/flash_attention/reset_attention_mask/adaptor.py $site_packages_dir/mindspeed/core/transformer/flash_attention/reset_attention_mask/adaptor.py
\cp $(dirname $0)/../patch/mindspeed.patch/2.2.0_core_r0.12.1/MindSpeed/mindspeed/features_manager/transformer/flash_attention/reset_attention_mask_feature.py $site_packages_dir/mindspeed/features_manager/transformer/flash_attention/reset_attention_mask_feature.py
echo -e "\033[32m Applied MindSpeed CP \033[0m"

# 取消容器的proxy。bug：待修复
unset https_proxy
unset http_proxy

unset HTTPS_PROXY
unset HTTP_PROXY

# 删除lock，防止卡住
find /root/.cache/torch_extensions -type f -name lock -delete

# 提前build
python3 -c "import mindspeed; from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder; RotaryPositionEmbeddingOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import MoeTokenPermuteOpBuilder; MoeTokenPermuteOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import GMMV2OpBuilder; GMMV2OpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder; MoeTokenUnpermuteOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import MatmulAddOpBuilder; MatmulAddOpBuilder().load()" &
python3 -c "import mindspeed; from mindspeed.op_builder import GroupMatmulAddOpBuilder; GroupMatmulAddOpBuilder().load()" &

#######################################
# 获取当前节点IP
echo $CURRENT_IP
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 备份脚本
  mkdir -p $JOB_LOG_DIR_CURR
  cp $(realpath $(dirname $DEFAULT_SH)/main.sh) "${JOB_LOG_DIR_CURR}/." # cp main.sh
  cp $(realpath $0) "${JOB_LOG_DIR_CURR}/."  # cp start.sh
  cp $(realpath $DEFAULT_SH) "${JOB_LOG_DIR_CURR}/." # cp run.sh

  # 主节点启动
  echo RAY_DEBUG=$RAY_DEBUG
  echo RAY_DEBUG_POST_MORTEM=$RAY_DEBUG_POST_MORTEM

  ray start --head --port 4918 --dashboard-host="0.0.0.0" --node-ip-address=$CURRENT_IP --dashboard-port=4919 --disable-usage-stats

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*(NPU|GPU))' | head -n 1)pwd
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -ge "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU/GPU resources), starting Python script."
          ray status
          nohup bash $DEFAULT_SH 2>&1 > $JOB_LOG_DIR_CURR/run.log &
          tail -f $JOB_LOG_DIR_CURR/run.log
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="${MASTER_ADDR}:4918" --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

echo "start.sh ended on ${CURRENT_IP}"
