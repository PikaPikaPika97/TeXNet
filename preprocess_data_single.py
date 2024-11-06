import numpy as np
import scipy.io as scio
import os

root = "/root/autodl-tmp/HADAR_database"

# 要处理的场景和文件代号
scene_id = [1]
heatcube_id = 6
################ eList Processing #######################
ids = [f"{_}_{heatcube_id:04d}" for _ in ["L", "R"]]
SUBFOLDERS = [f"Scene{s_id}" for s_id in scene_id]

print("eMap preprocessing")
for subfolder in SUBFOLDERS:
    print("Processing subfolder", subfolder)
    e_files = []
    elist_file = os.path.join(root, subfolder, "GroundTruth", "eMap", "eList.mat")
    for id in ids:
        e_files.append(
            os.path.join(root, subfolder, "GroundTruth", "eMap", f"eMap_{id}.mat")
        )
    for i in range(2):
        e_list = np.squeeze(np.asarray(scio.loadmat(elist_file)["eList"]))
        e_data = scio.loadmat(e_files[i])["eMap"]

        data = e_list[e_data - 1] - 1
        if i == 0:
            np.save(
                os.path.join(
                    root,
                    subfolder,
                    "GroundTruth",
                    "eMap",
                    f"new_eMap_L_{heatcube_id:04d}.npy",
                ),
                np.asarray(data),
            )
        else:
            np.save(
                os.path.join(
                    root,
                    subfolder,
                    "GroundTruth",
                    "eMap",
                    f"new_eMap_R_{heatcube_id:04d}.npy",
                ),
                np.asarray(data),
            )

################### S_beta Processing ###################################################
import torch.nn.functional as F
import torch

ids = [f"{_}_{heatcube_id:04d}" for _ in ["L", "R"]]
SUBFOLDERS = [f"Scene{s_id}" for s_id in scene_id]

print("S_beta preprocessing")
for subfolder in SUBFOLDERS:
    S_files = []
    print("Processing subfolder", subfolder)
    for id in ids:
        S_files.append(os.path.join(root, subfolder, "HeatCubes", f"{id}_heatcube,mat"))

    for i in range(2):
        img = torch.tensor(np.asarray(scio.loadmat(S_files[i])["S"]))
        img = torch.permute(img, (2, 0, 1))
        img = torch.reshape(img, (1, 10, 800, 1000))
        [b, c, h, w] = img.shape
        quadratic_split = F.avg_pool2d(img, (h // 2, w))
        mean = quadratic_split.numpy()

        if i == 0:
            np.save(
                os.path.join(
                    root,
                    subfolder,
                    "HeatCubes",
                    f"S_EnvObj_L_{heatcube_id:04d}.npy",
                ),
                np.asarray(mean),
            )
        else:
            np.save(
                os.path.join(
                    root,
                    subfolder,
                    "HeatCubes",
                    f"S_EnvObj_R_{heatcube_id:04d}.npy",
                ),
                np.asarray(mean),
            )
