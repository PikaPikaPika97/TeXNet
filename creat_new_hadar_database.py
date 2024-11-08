# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///
import os
import scipy.io as sio
import numpy as np


class HadarDatabaseProcessor:
    """处理Hadar数据库
    
    该类用于处理Hadar数据库，包括合成数据和实验数据。合成数据包括Scene1~Scene10，实验数据包括Scene11。合成数据中的GroundTruth文件夹下有Depth、eMap、resMap、TeX、tMap和vMap总共6个子文件夹，每个子文件夹下有10个.mat文件，与HeatCubes文件夹中的10个.mat文件一一对应（L和R各5帧）。HeatCubes文件夹下有5帧的heatcube.mat文件，每一帧有R和L两个文件。实验数据中的GroundTruth文件夹下有xMap、eMap、resMap、TeX、tMap和vMap总共6个子文件夹，每个子文件夹下有8个.mat文件，与HeatCubes文件夹中的8个.mat文件一一对应（L和R各4帧）。HeatCubes文件夹下有4帧的heatcube.mat文件，每一帧有R和L两个文件。Radiance_EnvObj文件夹下有四个.mat文件，对应4帧图像的环境目标辐射，命名格式为"S_EnvObj_0001.mat"至"S_EnvObj_0004.mat"，尺寸为2*49，分别对应于两个环境对象的不同波段的辐射值。
    """
    def __init__(
        self,
        database_path: str,
        target_size: list[list[int], list[int]],
        target_channel: list[int],
    ):
        self.path = database_path
        self.target_size = target_size
        self.target_height, self.target_width = target_size
        self.target_channel = target_channel
        self.exp_target_channel = [i - 4 for i in self.target_channel]

    def process_synthetic_GroundTruth(self, GroundTruth_path: str):
        """处理synthetic dateset场景中的GroundTruth文件夹

        处理一个场景中的GroundTruth文件夹，该文件夹下有Depth、eMap、resMap、TeX、tMap和vMap总共6个子文件夹，每个子文件夹下有10个.mat文件，与HeatCubes文件夹中的10个.mat文件一一对应（L和R各5帧）。每个子文件夹下的.mat文件的命名方式为"folder_L_000X.mat"或"folder_R_000X.mat"，X为帧序号，folder为.mat文件的父文件夹名。将每个.mat文件的尺寸调整为target_size，调整通道数，然后保存覆盖原始文件。

        Args:
            GroundTruth_path (str): GroundTruth文件夹路径

        """
        print(f"Processing GroundTruth in {GroundTruth_path}")
        # 生成Depth、eMap、resMap、TeX、tMap和vMap文件夹路径
        subfolders: list[str] = [
            os.path.join(GroundTruth_path, folder)
            for folder in ["Depth", "eMap", "resMap", "tMap", "vMap"]
        ]
        # 遍历Depth、eMap、resMap、TeX、tMap和vMap文件夹
        for subfolder in subfolders:
            print(f"Processing {subfolder}")
            # 生成第1~5帧的.mat文件路径，每一帧有R和L两个文件。
            # 获取文件夹名
            folder_name = os.path.basename(subfolder)
            # 生成第1~5帧的.mat文件路径，每一帧有R和L两个文件
            files: list[str] = [
                os.path.join(subfolder, f"{folder_name}_{direction}_000{i}.mat")
                for i in range(1, 6)
                for direction in ["L", "R"]
            ]
            # 遍历第1~5帧的.mat文件
            # 获取.mat文件中变量的名字，注意只有Depth文件夹下的.mat文件中变量名为"depth，其他文件夹下的变量名与文件夹名相同
            variable_name = "depth" if folder_name == "Depth" else folder_name
            for file in files:
                # 读取.mat文件
                data = sio.loadmat(file)[variable_name]
                # 使用np.ix_创建索引网格，用于调整.mat文件的尺寸，保留通道数不变
                if len(data.shape) == 2:
                    data = data[np.ix_(self.target_height, self.target_width)]
                elif len(data.shape) == 3:
                    data = data[
                        np.ix_(
                            self.target_height, self.target_width, range(data.shape[2])
                        )
                    ]
                # 保存调整后的.mat文件（覆盖原始文件）
                sio.savemat(file, {variable_name: data})
            print(f"{subfolder} processed successfully")
        print(f"GroundTruth in {GroundTruth_path} processed successfully")

    def process_synthetic_HeatCubes(self, HeatCubes_path: str):
        """处理synthetic dateset场景中的HeatCubes文件夹

        HeatCubes文件夹下有5帧的heatcube.mat文件，每一帧有R和L两个文件。文件命名的格式为"L_000X_heatcube.mat"或"R_000X_heatcube.mat"，X为帧序号。将每个heatcube.mat文件的尺寸调整为target_size，并保留选择的通道数，然后保存覆盖原始文件。

        Args:
            HeatCubes_path (str): HeatCubes文件夹路径

        """
        print(f"Processing HeatCubes in {HeatCubes_path}")
        # 生成第1~5帧的heatcube.mat文件路径，每一帧有R和L两个文件。
        # 文件命名的格式为"L_000X_heatcube.mat"和"R_000X_heatcube.mat"，X为帧序号
        heatcubes_files: list[str] = [
            os.path.join(HeatCubes_path, f"{direction}_000{i}_heatcube.mat")
            for i in range(1, 6)
            for direction in ["L", "R"]
        ]
        # 遍历第1~5帧的heatcube.mat文件，原始.mat文件的尺寸为1080*1920*54
        for heatcube_file in heatcubes_files:
            # 读取heatcube.mat文件
            heatcube = sio.loadmat(heatcube_file)["S"]

            # 使用np.ix_创建索引网格，用于调整heatcube的尺寸
            heatcube = heatcube[
                np.ix_(self.target_height, self.target_width, self.target_channel)
            ]
            # 保存调整后的heatcube.mat文件（覆盖原始文件）
            sio.savemat(heatcube_file, {"S": heatcube})
        print("HeatCubes processed successfully")

    def process_experimental_GroundTruth(self, GroundTruth_path: str):
        """处理experimental dateset场景中的GroundTruth文件夹

        处理Scene11中的GroundTruth文件夹，该文件夹下有xMap、eMap、resMap、TeX、tMap和vMap总共6个子文件夹，每个子文件夹下有8个.mat文件，与HeatCubes文件夹中的8个.mat文件一一对应（L和R各4帧）。每个子文件夹下的.mat文件的命名方式为"folder_L_000X.mat"或"folder_R_000X.mat"，X为帧序号，folder为.mat文件的父文件夹名。将每个.mat文件的通道数调整为exp_target_channel，保持高度和宽度不变，然后保存覆盖原始文件。
        Args:
            GroundTruth_path (str): GroundTruth文件夹路径

        """
        print(f"Processing GroundTruth in {GroundTruth_path}")
        # 生成Depth、eMap、resMap、TeX、tMap和vMap文件夹路径
        subfolders: list[str] = [
            os.path.join(GroundTruth_path, folder)
            for folder in ["xMap", "eMap", "resMap", "tMap", "vMap"]
        ]
        # 遍历eMap、resMap、TeX、tMap和vMap文件夹
        for subfolder in subfolders:
            print(f"Processing {subfolder}")
            # 生成第1~4帧的.mat文件路径，每一帧有R和L两个文件。
            # 获取文件夹名
            folder_name = os.path.basename(subfolder)
            # 生成第1~4帧的.mat文件路径，每一帧有R和L两个文件
            files: list[str] = [
                os.path.join(subfolder, f"{folder_name}_{direction}_000{i}.mat")
                for i in range(1, 5)
                for direction in ["L", "R"]
            ]
            # 遍历第1~4帧的.mat文件
            # 获取.mat文件中变量的名字，注意只有Depth文件夹下的.mat文件中变量名为"depth，其他文件夹下的变量名与文件夹名相同
            variable_name = "depth" if folder_name == "Depth" else folder_name
            for file in files:
                # 读取.mat文件
                data = sio.loadmat(file)[variable_name]
                # 使用np.ix_创建索引网格，用于调整.mat文件的尺寸，保留通道数不变
                if len(data.shape) == 2:
                    data = data[np.ix_(range(260), range(1500))]
                elif len(data.shape) == 3:
                    data = data[np.ix_(range(260), range(1500), range(data.shape[2]))]
                # 保存调整后的.mat文件（覆盖原始文件）
                sio.savemat(file, {variable_name: data})
            print(f"{subfolder} processed successfully")
        print(f"GroundTruth in {GroundTruth_path} processed")

    def process_experimental_HeatCubes(self, HeatCubes_path: str):
        """处理experimental dateset场景中的HeatCubes文件夹

        处理方式与处理synthetic dateset场景中的HeatCubes文件夹相同，但是要注意只有4帧图像，每帧图像有R和L两个文件，文件命名的格式为"L_000X_heatcube.mat"或"R_000X_heatcube.mat"，X为帧序号。每个.mat文件的尺寸为260*1500*49，对应于合成场景的第5至第53波段。

        Args:
            HeatCubes_path (str): HeatCubes文件夹路径

        """
        print(f"Processing HeatCubes in {HeatCubes_path}")
        # 生成第1~4帧的heatcube.mat文件路径，每一帧有R和L两个文件。
        # 文件命名的格式为"L_000X_heatcube.mat"和"R_000X_heatcube.mat"，X为帧序号
        heatcubes_files: list[str] = [
            os.path.join(HeatCubes_path, f"{direction}_000{i}_heatcube.mat")
            for i in range(1, 5)
            for direction in ["L", "R"]
        ]
        # 遍历第1~4帧的heatcube.mat文件，原始.mat文件的尺寸为260*1500*49
        for heatcube_file in heatcubes_files:
            # 读取heatcube.mat文件
            heatcube = sio.loadmat(heatcube_file)["HSI"]
            # 使用np.ix_创建索引网格，用于调整heatcube的尺寸，只更改通道数，而不改变高度和宽度
            heatcube = heatcube[
                np.ix_(range(260), range(1500), self.exp_target_channel)
            ]
            # 保存调整后的heatcube.mat文件（覆盖原始文件）
            sio.savemat(heatcube_file, {"HSI": heatcube})
        print("HeatCubes processed successfully")

    def process_experimental_Radiance_EnvObj(self, Radiance_EnvObj_path: str):
        """处理Scene11下的Radiance_EnvObj文件夹

        Scnen11下还有一个Radiance_EnvObj子文件夹，其中有四个.mat文件，对应4帧图像的环境目标辐射，命名格式为"S_EnvObj_0001.mat"至"S_EnvObj_0004.mat"，尺寸为2*49，分别对应于两个环境对象的不同波段的辐射值。将每个.mat文件的尺寸调整为2*target_channel，并保存覆盖原始文件。

        Args:
            Radiance_EnvObj_path (str): Radiance_EnvObj文件夹路径

        """
        print(f"Processing Radiance_EnvObj in {Radiance_EnvObj_path}")
        # 生成第1~4帧的S_EnvObj_000X.mat文件路径
        files: list[str] = [
            os.path.join(Radiance_EnvObj_path, f"S_EnvObj_000{i}.mat")
            for i in range(1, 5)
        ]
        # 遍历第1~4帧的S_EnvObj_000X.mat文件
        for file in files:
            # 读取.mat文件
            data = sio.loadmat(file)["S_EnvObj"]
            # 使用np.ix_创建索引网格，用于调整.mat文件的尺寸，保留通道数不变
            data = data[np.ix_(range(2), self.exp_target_channel)]
            # 保存调整后的.mat文件（覆盖原始文件）
            sio.savemat(file, {"S_EnvObj": data})
        print("Radiance_EnvObj processed successfully")

    def process_hadar_database(self):
        """处理hadar数据库"""
        # 首先处理合成数据
        print("Processing synthetic dataset")
        # 生成Scene1~Scene10的文件夹路径
        Scene_folders: list = [
            os.path.join(self.path, f"Scene{i}") for i in range(1, 11)
        ]
        # 遍历Scene1~Scene10文件夹
        for Scene_folder in Scene_folders:
            # 生成GroundTruth文件夹路径
            GroundTruth_path = os.path.join(Scene_folder, "GroundTruth")
            # 生成HeatCubes文件夹路径
            HeatCubes_path = os.path.join(Scene_folder, "HeatCubes")
            # 处理GroundTruth文件夹
            self.process_synthetic_GroundTruth(GroundTruth_path)
            # 处理HeatCubes文件夹
            self.process_synthetic_HeatCubes(HeatCubes_path)
        print("Synthetic dataset processed successfully")

        # 然后处理实验数据
        print("Processing experimental dataset")
        # 生成Scene11文件夹路径
        Scene11_folder = os.path.join(self.path, "Scene11")
        # 生成GroundTruth文件夹路径
        GroundTruth_path = os.path.join(Scene11_folder, "GroundTruth")
        # 生成HeatCubes文件夹路径
        HeatCubes_path = os.path.join(Scene11_folder, "HeatCubes")
        # 生成Radiance_EnvObj文件夹路径
        Radiance_EnvObj_path = os.path.join(Scene11_folder, "Radiance_EnvObj")
        # 处理GroundTruth文件夹
        self.process_experimental_GroundTruth(GroundTruth_path)
        # 处理HeatCubes文件夹
        self.process_experimental_HeatCubes(HeatCubes_path)
        # 处理Radiance_EnvObj文件夹
        self.process_experimental_Radiance_EnvObj(Radiance_EnvObj_path)
        print("Experimental dataset processed successfully")


if __name__ == "__main__":
    database_path = r"D:\DDownload\HADAR\OneDrive_2023-09-20\HADAR_database_new"
    target_height = list(range(800))
    target_width = list(range(1000))
    target_channel = [5, 10, 20, 30, 34, 35, 36, 37, 43, 46]
    # 创建HadarDatabaseProcessor对象
    processor = HadarDatabaseProcessor(
        database_path=database_path,
        target_size=[target_height, target_width],
        target_channel=target_channel,
    )
    # 处理hadar数据库
    processor.process_hadar_database()

