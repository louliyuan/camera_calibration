import pandas as pd


def read_image_point_data(filename) :
    """
    读取三维控制场像片的像点坐标
    :param filename: 文件名
    :return: 像点数据字典 {点号: (x, y)}
    """
    image_points = pd.read_csv(
        filename,
        sep='\s+', header=None, names=["image_pid", "x", "y"]
    )
    image_pid = image_points["image_pid"].values
    x = image_points["x"].values
    y = image_points["y"].values
    return image_pid, x, y

def read_control_point_data(filename):
    """
    读取物方控制点三维坐标
    :param filename: 文件名
    :return: 控制点数据字典 {点号: (x, y, z)}
    """
    control_points = pd.read_csv(
        filename,
        sep='\s+', header=None, names=["wu_id", "X", "Y", "Z"],
        skiprows=1 #跳过第一行
    )
    wu_id = control_points["wu_id"].values
    X = control_points["X"].values
    Y = control_points["Y"].values
    Z = control_points["Z"].values
    point_count = len(control_points)
    return wu_id, X, Y, Z, point_count