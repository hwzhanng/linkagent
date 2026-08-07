# 控制环境

环境由 `environment.yml` 重建，不保存 `.conda/`：

```bash
cd /home/descfly/linkagent/botSDK/linkerhand-control-workspace
conda env create --prefix .conda/envs/linkerhand --file environment.yml
source scripts/activate_env.sh
```

最小直接控制依赖包括 Python 3.11、NumPy、PyYAML、pyserial、python-can、
python-can-candle、minimalmodbus 和 pymodbus。系统侧需要 `can-utils`。

ROS 2 依赖单独构建到 `ros2_official/`，不装入该 Conda 环境。
