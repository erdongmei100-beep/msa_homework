# Sioux Falls TAP (MSA/MSWA)

本项目实现基于 Sioux Falls 网络的静态交通分配 (TAP) 示例，使用 MSA 以及加权步长的 MSWA。核心计算基于 NumPy 向量化、SciPy 稀疏图最短路以及 Numba 加速的全有全无分配。

## 目录结构

- `data/`：Sioux Falls 网络与 OD 数据
- `src/`：源代码
  - `tntp_io.py`：TNTP/CSV 数据读取
  - `msa_core.py`：MSA/MSWA 求解逻辑
  - `main.py`：示例入口
- `requirements.txt`：依赖列表

## 准备数据

项目预期的文件路径：
- `data/SiouxFalls_net.tntp`
- `data/SiouxFalls_od.csv`

如网络受限，请手动从 [bstabler/TransportationNetworks](https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls) 下载上述文件并放入 `data/` 目录。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
python -m src.main
```

运行后会打印每次迭代的步长与相对间隙，最终输出收敛信息和总流量。
