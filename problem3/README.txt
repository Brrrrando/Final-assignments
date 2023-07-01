colmap下载地址：https://github.com/colmap/colmap/releases
1、先在上面的地址下载colmap
2、将收集好的数据集放在nerf-pytorch\data\nerf_llff_data\CrystalIO\images该目录下
3、利用colmap对数据集提取特征点并输出结果、配准并重建
4、运行nerf模型（run_nerf.py）训练并输出重建结果，结果保存在nerf-pytorch\logs\CrystalIO目录下