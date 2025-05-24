import numpy as np

"""
@brief: 相机标定误差方程
@input: 像素坐标 x, y
    •  世界坐标 X, Y, Z
    •  像素与世界点的 ID 对应关系 id, pt_wu
    •  相机参数（焦距、主点、畸变系数）和相机姿态（R, Xs, Ys, Zs）
@output:
    •   误差方程系数矩阵 B
    •  常数项 L
    •  附加约束矩阵 C 和其常数项 W2（用于正则化旋转矩阵 R）
"""
def calibration_SetErrorEquation(image_id, x, y, pt_wu, X, Y, Z, Xs, Ys, Zs, R, x0, y0, fx, fy, k1, k2, p1, p2):
    number = len(x)
    B_list = []  # 使用列表存储，然后一次性转换为numpy数组，效率更高
    L_list = []

    for i in range(number):
        if image_id[i] is not None:
            # 匹配
            row_indices = np.where(np.array(pt_wu) == image_id[i])[0]
            if len(row_indices) > 0:
                row = row_indices[0]  # 取第一个匹配的行

                # 计算相机坐标系下的坐标
                X_cam = X[row] - Xs
                Y_cam = Y[row] - Ys
                Z_cam = Z[row] - Zs

                XX = R[0, 0] * X_cam + R[0, 1] * Y_cam + R[0, 2] * Z_cam
                YY = R[1, 0] * X_cam + R[1, 1] * Y_cam + R[1, 2] * Z_cam
                ZZ = R[2, 0] * X_cam + R[2, 1] * Y_cam + R[2, 2] * Z_cam

                # 检查 ZZ 是否过小，避免除以零或数值不稳定
                if abs(ZZ) < 1e-9:  # 这个阈值可以根据实际情况调整
                    # print(f"Warning: ZZ is close to zero for image_id {image_id[i]}, point {pt_wu[row]}. Skipping.")
                    continue

                # 计算畸变相关量
                _x_centered = x[i] - x0
                _y_centered = y[i] - y0
                _r_squared_pixel = _x_centered**2 + _y_centered**2

                delta_x = _x_centered * (k1 * _r_squared_pixel + k2 * (_r_squared_pixel**2)) + \
                          p1 * (_r_squared_pixel + 2 * _x_centered**2) + \
                          2 * p2 * _x_centered * _y_centered
                delta_y = _y_centered * (k1 * _r_squared_pixel + k2 * (_r_squared_pixel**2)) + \
                          p2 * (_r_squared_pixel + 2 * _y_centered**2) + \
                          2 * p1 * _x_centered * _y_centered

                # 校正畸变后的像点坐标 -->相对于主点 x0, y0
                x_corr = x[i] - x0 - delta_x
                y_corr = y[i] - y0 - delta_y

                # 误差方程常数项 L
                L_list.append([x_corr + fx * XX / ZZ])
                L_list.append([y_corr + fy * YY / ZZ])


                # 误差方程系数矩阵 B
                B_row1 = [
                    (R[0, 0] * fx + R[2, 0] * x_corr) / ZZ,
                    (R[0, 1] * fx + R[2, 1] * x_corr) / ZZ,
                    (R[0, 2] * fx + R[2, 2] * x_corr) / ZZ,
                    -fx * X_cam / ZZ,                       # d/dR00
                    0,                                      # d/dR01
                    -x_corr * X_cam / ZZ,                   # d/dR02
                    -fx * Y_cam / ZZ,                       # d/dR10
                    0,                                      # d/dR11
                    -x_corr * Y_cam / ZZ,                   # d/dR12
                    -fx * Z_cam / ZZ,                       # d/dR20
                    0,                                      # d/dR21
                    -x_corr * Z_cam / ZZ,                   # d/dR22
                    1 - k1 * _r_squared_pixel - k2 * _r_squared_pixel ** 2 - _x_centered * (2 * k1 * _x_centered + 4 * k2 * _x_centered * _r_squared_pixel + 4 * p1 * _x_centered + 2 * p2 * _y_centered), # d/dx0
                    -(_x_centered * (2 * k1 * _y_centered + 4 * k2 * _y_centered * _r_squared_pixel) + p1 * (2 * _x_centered * _y_centered) + p2 * (_r_squared_pixel + 2 * _x_centered**2 + 2 * _y_centered * _y_centered) ), # d/dy0
                    XX / ZZ,                                # d/dfx
                    0,                                      # d/dfy
                    _x_centered * _r_squared_pixel,         # d/dk1
                    _x_centered * (_r_squared_pixel ** 2),  # d/dk2
                    (3 * _x_centered**2 + _y_centered**2),  # d/dp1
                    2 * _x_centered * _y_centered           # d/dp2
                ]
                B_row2 = [
                    (R[1, 0] * fy + R[2, 0] * y_corr) / ZZ,  # d/dXs
                    (R[1, 1] * fy + R[2, 1] * y_corr) / ZZ,  # d/dYs
                    (R[1, 2] * fy + R[2, 2] * y_corr) / ZZ,  # d/dZs
                    0,                                      # d/dR00
                    -fy * X_cam / ZZ,                       # d/dR01
                    -y_corr * X_cam / ZZ,                   # d/dR02
                    0,                                      # d/dR10
                    -fy * Y_cam / ZZ,                       # d/dR11
                    -y_corr * Y_cam / ZZ,                   # d/dR12
                    0,                                      # d/dR20
                    -fy * Z_cam / ZZ,                       # d/dR21
                    -y_corr * Z_cam / ZZ,                   # d/dR22
                   -(_y_centered * (2 * k1 * _x_centered + 4 * k2 * _x_centered * _r_squared_pixel) + p1 * (_r_squared_pixel + 2 * _x_centered**2 + 2*_y_centered**2 ) + p2 * (2 * _x_centered * _y_centered)), #d/dx0 (此项来自您代码中B_row2[12]的结构)
                    1 - k1 * _r_squared_pixel - k2 * _r_squared_pixel ** 2 - _y_centered * (2 * k1 * _y_centered + 4 * k2 * _y_centered * _r_squared_pixel + 4 * p2 * _y_centered + 2 * p1 * _x_centered), # d/dy0
                    0,                                      # d/dfx
                    YY / ZZ,                                # d/dfy
                    _y_centered * _r_squared_pixel,         # d/dk1
                    _y_centered * (_r_squared_pixel ** 2),  # d/dk2
                    2 * _x_centered * _y_centered,          # d/dp1
                    (3 * _y_centered**2 + _x_centered**2)   # d/dp2
                ]
                B_list.append(B_row1)
                B_list.append(B_row2)

    if not B_list: # 如果没有有效的点对
        # 返回空的或适当维度的零矩阵，并可能引发错误或警告
        num_params = 20 # Xs,Ys,Zs, R(9), x0,y0,fx,fy,k1,k2,p1,p2
        B = np.empty((0, num_params))
        L = np.empty((0, 1))
    else:
        B = np.array(B_list)
        L = np.array(L_list)


    # 附加约束矩阵 C (用于旋转矩阵 R 的正交性)
    # C1 对应旋转矩阵的参数 (9个 R_ij)
    C1 = np.array([
        [0, 0, 0, 2 * R[0, 0], 2 * R[0, 1], 2 * R[0, 2], 0, 0, 0, 0, 0, 0],  # 12 列
        [0, 0, 0, 0, 0, 0, 2 * R[1, 0], 2 * R[1, 1], 2 * R[1, 2], 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2 * R[2, 0], 2 * R[2, 1], 2 * R[2, 2]],
        [0, 0, 0, R[0, 1], R[0, 0], 0, R[1, 1], R[1, 0], 0, R[2, 1], R[2, 0], 0],
        [0, 0, 0, R[0, 2], 0, R[0, 0], R[1, 2], 0, R[1, 0], R[2, 2], 0, R[2, 0]],
        [0, 0, 0, 0, R[0, 2], R[0, 1], 0, R[1, 2], R[1, 1], 0, R[2, 2], R[2, 1]]
    ])
    C_suffix_for_internal = np.zeros((6,8)) # x0,y0,fx,fy,k1,k2,p1,p2
    C = np.hstack(( C1, C_suffix_for_internal))


    # 附加约束常数项 W2
    W2 = -np.array([
        [R[0,0]**2 + R[0,1]**2 + R[0,2]**2 - 1.0],  # row0.row0 - 1
        [R[1,0]**2 + R[1,1]**2 + R[1,2]**2 - 1.0],  # row1.row1 - 1
        [R[2,0]**2 + R[2,1]**2 + R[2,2]**2 - 1.0],  # row2.row2 - 1
        [R[0,0]*R[0,1] + R[1,0]*R[1,1] + R[2,0]*R[2,1]],      # col0.col1
        [R[0,0]*R[0,2] + R[1,0]*R[1,2] + R[2,0]*R[2,2]],      # col0.col2
        [R[0,1]*R[0,2] + R[1,1]*R[1,2] + R[2,1]*R[2,2]]       # col1.col2
    ])

    # 检查 W2 是否包含无效值 (理论上不应再发生，除非R本身包含NaN/inf)
    if np.any(np.isnan(W2)) or np.any(np.isinf(W2)):
        print("DEBUG: R matrix causing W2 issues:\n", R)
        raise ValueError("W2 包含无效值 (NaN 或 inf)。请检查 R 矩阵。")

    return B, L, C, W2