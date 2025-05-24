import numpy as np
# from calibration import calibration_SetErrorEquation
from Read_data import read_control_point_data, read_image_point_data
from math import sin, cos, atan, asin, pi,fabs,sqrt
from Matrix_computation import compute_rotation_matrix, transpose_matrix, multiply_matrix, add_matrix, inverse_matrix
import os

def normalize_image_points(image_pid, x, y, control_points_map):
    """
    归一化像点坐标并筛选有效点
    """
    x_norm = x - 2672
    y_norm = 2004 - y
    valid_pids, valid_x, valid_y = [], [], []
    for i, pid in enumerate(image_pid):
        if pid in control_points_map:
            valid_pids.append(pid)
            valid_x.append(x_norm[i])
            valid_y.append(y_norm[i])
    return valid_pids, valid_x, valid_y

def initialize_parameters(X, Y, Z, num_images):
    """
    初始化内外方位元素
    """
    x0, y0, f = 0.0, 0.0, 60.0
    k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0
    Xs_list = [np.mean(X)] * num_images
    Ys_list = [np.mean(Y)] * num_images
    Zs_list = [np.mean(Z) + 200.0] * num_images
    phi_list = [0.0] * num_images
    omega_list = [0.0] * num_images
    kappa_list = [0.0] * num_images
    return x0, y0, f, k1, k2, p1, p2, Xs_list, Ys_list, Zs_list, phi_list, omega_list, kappa_list

def compute_residuals_and_jacobian(x_img, y_img, X_cp, Y_cp, Z_cp, Xs, Ys, Zs, phi, omega, kappa, x0, y0, f, k1, k2, p1, p2):
    """
    计算残差和雅可比矩阵
    """
    rr_sq = (x_img - x0) ** 2 + (y_img - y0) ** 2
    R, a1, a2, a3, b1, b2, b3, c1, c2, c3 = compute_rotation_matrix(phi, omega, kappa)
    Xp = a1 * (X_cp - Xs) + b1 * (Y_cp - Ys) + c1 * (Z_cp - Zs)
    Yp = a2 * (X_cp - Xs) + b2 * (Y_cp - Ys) + c2 * (Z_cp - Zs)
    Zp = a3 * (X_cp - Xs) + b3 * (Y_cp - Ys) + c3 * (Z_cp - Zs)

    if fabs(Zp) < 1e-6:
        raise ValueError("Zp is near zero, skipping point.")

    deltax = (x_img - x0) * (k1 * rr_sq + k2 * rr_sq ** 2) + p1 * (rr_sq + 2 * (x_img - x0) ** 2) + 2 * p2 * (x_img - x0) * (y_img - y0)
    deltay = (y_img - y0) * (k1 * rr_sq + k2 * rr_sq ** 2) + p2 * (rr_sq + 2 * (y_img - y0) ** 2) + 2 * p1 * (x_img - x0) * (y_img - y0)

    L = np.zeros((2, 1))
    L[0, 0] = x_img - (x0 - f * Xp / Zp - deltax)
    L[1, 0] = y_img - (y0 - f * Yp / Zp - deltay)

    A = np.zeros((2, 13))
    A[0, 0] = (a1 * f + a3 * (x_img - x0)) / Zp
    A[0, 1] = (b1 * f + b3 * (x_img - x0)) / Zp
    A[0, 2] = (c1 * f + c3 * (x_img - x0)) / Zp
    A[0, 3] = (y_img - y0) * sin(omega) - ((x_img - x0) * ((x_img - x0) * cos(kappa) - (y_img - y0) * sin(kappa)) / f + f * cos(kappa)) * cos(omega)
    A[0, 4] = -f * sin(kappa) - (x_img - x0) * ((x_img - x0) * sin(kappa) + (y_img - y0) * cos(kappa)) / f
    A[0, 5] = (y_img - y0)
    A[0, 6] = (x_img - x0) / f
    A[0, 7] = 1.0
    A[0, 8] = 0.0
    A[0, 9] = -(x_img - x0) * rr_sq
    A[0, 10] = -(x_img - x0) * rr_sq ** 2
    A[0, 11] = -(rr_sq + 2 * (x_img - x0) ** 2)
    A[0, 12] = -2 * (x_img - x0) * (y_img - y0)

    A[1, 0] = (a2 * f + a3 * (y_img - y0)) / Zp
    A[1, 1] = (b2 * f + b3 * (y_img - y0)) / Zp
    A[1, 2] = (c2 * f + c3 * (y_img - y0)) / Zp
    A[1, 3] = -(x_img - x0) * sin(omega) - ((y_img - y0) * ((x_img - x0) * cos(kappa) - (y_img - y0) * sin(kappa)) / f - f * sin(kappa)) * cos(omega)
    A[1, 4] = -f * cos(kappa) - (y_img - y0) * ((x_img - x0) * sin(kappa) + (y_img - y0) * cos(kappa)) / f
    A[1, 5] = -(x_img - x0)
    A[1, 6] = (y_img - y0) / f
    A[1, 7] = 0.0
    A[1, 8] = 1.0
    A[1, 9] = -(y_img - y0) * rr_sq
    A[1, 10] = -(y_img - y0) * rr_sq ** 2
    A[1, 11] = -2 * (x_img - x0) * (y_img - y0)
    A[1, 12] = -(rr_sq + 2 * (y_img - y0) ** 2)

    return L, A

def back_intersection(image_points_files, control_points_file):
    """
    后方交会函数-多影像平差
    """
    # 读取物方控制点数据
    wu_id, X, Y, Z, point_count = read_control_point_data(control_points_file)
    control_points_map = {wu_id[i]: (X[i], Y[i], Z[i]) for i in range(point_count)}

    # 读取像点数据
    all_image_data = []
    for image_points_file in image_points_files:
        image_pid, x, y = read_image_point_data(image_points_file)
        valid_pids, valid_x, valid_y = normalize_image_points(image_pid, x, y, control_points_map)
        all_image_data.append({'pids': valid_pids, 'x': valid_x, 'y': valid_y, 'count': len(valid_pids)})

    """
        @brief 内外方位元素初始化
        @note x0, y0, f, k1, k2, p1, p2是6张照片数据集共享的
        @note Xs, Ys, Zs, phi, omega, kappa是每张照片单独的
    """
    num_images = len(image_points_files)
    x0, y0, f, k1, k2, p1, p2, Xs_list, Ys_list, Zs_list, phi_list, omega_list, kappa_list = initialize_parameters(X, Y, Z, num_images)


    """
    @brief 矩阵初始化
    """
    num_extrinsic_params_per_image = 6
    num_intrinsic_params = 7  # f, x0, y0, k1, k2, p1, p2
    num_unknowns = num_extrinsic_params_per_image * num_images + num_intrinsic_params
    N_global = np.zeros((num_unknowns, num_unknowns))
    U_global = np.zeros((num_unknowns, 1))
    XX_corrections = np.full((num_unknowns, 1), 0.1)

    iteration = 0
    MAX_ITERATIONS = 100

    # 迭代条件
    LL = np.zeros(2 * point_count)
    Qii = np.zeros(13)
    ATA1, ATL1 = np.zeros((13, 13)), np.zeros((13, 1))
    Q = 0.0
    iteration = 0

    while True:
        N_global.fill(0)
        U_global.fill(0)
        all_L_values_iter = []

        for img_idx in range(num_images):
            current_image_data = all_image_data[img_idx]
            if current_image_data['count'] == 0:  # Skip if no valid points for this image
                continue

            Xs_curr = Xs_list[img_idx]
            Ys_curr = Ys_list[img_idx]
            Zs_curr = Zs_list[img_idx]
            phi_curr = phi_list[img_idx]
            omega_curr = omega_list[img_idx]
            kappa_curr = kappa_list[img_idx]

            # 计算旋转矩阵
            R_curr, a1, a2, a3, b1, b2, b3, c1, c2, c3 = compute_rotation_matrix(phi_curr, omega_curr, kappa_curr)

            for pt_idx in range(current_image_data['count']):
                x_img = current_image_data['x'][pt_idx]
                y_img = current_image_data['y'][pt_idx]
                pid_img = current_image_data['pids'][pt_idx]

                X_cp, Y_cp, Z_cp = control_points_map[pid_img]

                try:
                    L_point, A_local_derivs = compute_residuals_and_jacobian(
                        x_img, y_img, X_cp, Y_cp, Z_cp,
                        Xs_curr, Ys_curr, Zs_curr,
                        phi_curr, omega_curr, kappa_curr,
                        x0, y0, f, k1, k2, p1, p2
                    )
                except ValueError:
                    continue

                A_point_global = np.zeros((2, num_unknowns))
                extrinsic_start_col = img_idx * num_extrinsic_params_per_image
                A_point_global[:,
                extrinsic_start_col: extrinsic_start_col + num_extrinsic_params_per_image] = A_local_derivs[:,
                                                                                             :num_extrinsic_params_per_image]
                intrinsic_start_col = num_extrinsic_params_per_image * num_images
                A_point_global[:, intrinsic_start_col: intrinsic_start_col + num_intrinsic_params] = A_local_derivs[:,
                                                                                                     num_extrinsic_params_per_image:]

                N_global += np.dot(A_point_global.T, A_point_global)
                U_global += np.dot(A_point_global.T, L_point)
                all_L_values_iter.extend(L_point.flatten())

        if not all_L_values_iter:  # No points processed in this iteration
            print("Error: No points were processed in the current iteration. Check data or initial parameters.")
            return

        try:
            N_global_inv = np.linalg.inv(N_global)
            XX_corrections = np.dot(N_global_inv, U_global)
        except np.linalg.LinAlgError:
            print(f"Error: Singular N_global matrix at iteration {iteration + 1}. Cannot compute inverse.")
            print("Consider checking data collinearity, initial values, or adding regularization.")
            return

        for img_idx in range(num_images):
            offset = img_idx * num_extrinsic_params_per_image
            Xs_list[img_idx] += XX_corrections[offset + 0, 0]
            Ys_list[img_idx] += XX_corrections[offset + 1, 0]
            Zs_list[img_idx] += XX_corrections[offset + 2, 0]
            phi_list[img_idx] += XX_corrections[offset + 3, 0]
            omega_list[img_idx] += XX_corrections[offset + 4, 0]
            kappa_list[img_idx] += XX_corrections[offset + 5, 0]

        intrinsic_offset = num_extrinsic_params_per_image * num_images
        f += XX_corrections[intrinsic_offset + 0, 0]
        x0 += XX_corrections[intrinsic_offset + 1, 0]
        y0 += XX_corrections[intrinsic_offset + 2, 0]
        k1 += XX_corrections[intrinsic_offset + 3, 0]
        k2 += XX_corrections[intrinsic_offset + 4, 0]
        p1 += XX_corrections[intrinsic_offset + 5, 0]
        p2 += XX_corrections[intrinsic_offset + 6, 0]

        iteration += 1
        print(f"--- Iteration {iteration} ---")
        max_correction = np.max(np.abs(XX_corrections))
        print(f"Max Correction: {max_correction:.2e}")
        print(f"Updated f: {f:.4f}, x0: {x0:.4f}, y0: {y0:.4f}")
        if num_images > 0 and all_image_data[0]['count'] > 0:  # Check if first image has points
            print(f"Updated Xs[0]: {Xs_list[0]:.3f}, Ys[0]: {Ys_list[0]:.3f}, Zs[0]: {Zs_list[0]:.3f}")
            print(
                f"Updated phi[0] (rad): {phi_list[0]:.6f}, omega[0] (rad): {omega_list[0]:.6f}, kappa[0] (rad): {kappa_list[0]:.6f}")

        if max_correction < 1e-7:  # Threshold for convergence
            print(f"\nConverged after {iteration} iterations. Maximum correction: {max_correction:.2e}")
            break
        if iteration >= MAX_ITERATIONS:
            print(
                f"\nMaximum iterations ({MAX_ITERATIONS}) reached. Max correction: {max_correction:.2e}. Check results.")
            break

    VTPV = sum(val ** 2 for val in all_L_values_iter)
    degrees_of_freedom = len(all_L_values_iter) - num_unknowns
    sigma0 = sqrt(VTPV / degrees_of_freedom) if degrees_of_freedom > 0 else 1.0

    print(f"\n--- Final Results ({iteration} iterations) ---")
    print(f"Converged after {iteration} iterations. Sigma0: {sigma0:.6f}")

    # Check if N_global_inv was successfully computed in the last iteration
    try:
        # N_global_reg = N_global + np.eye(num_unknowns) * 1e-9 # Recalculate N_global if needed or use stored one
        # Qxx = np.linalg.inv(N_global_reg)
        Qxx = np.linalg.inv(N_global)  # Use N_global from last successful iteration
        param_std_devs = sigma0 * np.sqrt(np.abs(np.diag(Qxx)))
    except np.linalg.LinAlgError:
        print(
            "Error: Could not compute N_global_inv for final standard deviations. Displaying parameters without std dev.")
        param_std_devs = np.full(num_unknowns, np.nan)

    print("\nShared Intrinsic Parameters:")
    intr_param_names = ["f", "x0", "y0", "k1", "k2", "p1", "p2"]
    intr_values = [f, x0, y0, k1, k2, p1, p2]
    for i in range(num_intrinsic_params):
        idx_in_xx = intrinsic_offset + i
        print(f"  {intr_param_names[i]:<3s}: {intr_values[i]:>12.6f} +/- {param_std_devs[idx_in_xx]:.6e}")

    print("\nExterior Orientation Parameters (per image):")
    extr_param_names = ["Xs", "Ys", "Zs", "phi", "omega", "kappa"]
    extr_units = ["m", "m", "m", "deg", "deg", "deg"]

    for img_idx in range(num_images):
        print(f"  Image {img_idx + 1}:")
        current_img_params = [
            Xs_list[img_idx], Ys_list[img_idx], Zs_list[img_idx],
            phi_list[img_idx], omega_list[img_idx], kappa_list[img_idx]
        ]
        for i in range(num_extrinsic_params_per_image):
            param_val = current_img_params[i]
            idx_in_xx = img_idx * num_extrinsic_params_per_image + i
            std_dev_val = param_std_devs[idx_in_xx]

            display_val = param_val
            display_std_dev = std_dev_val
            unit = extr_units[i]

            if extr_param_names[i] in ['phi', 'omega', 'kappa']:
                display_val = param_val * 180.0 / pi
                display_std_dev = std_dev_val * 180.0 / pi

            print(f"    {extr_param_names[i]:<7s}: {display_val:>12.6f} +/- {display_std_dev:.6e} {unit}")

    return

# 调用函数
if __name__ == "__main__":
    base_path = '/Users/louliyuan/PycharmProjects/PythonProject/Camera_Check/相机检校数据/'

    # Construct full paths for image files
    image_file_names = [
        '2-40-g11-k1-001.txt',
        '2-40-g11-k1-002.txt',
        '2-40-g11-k1-003.txt',
        '2-40-g11-k1-004.txt',
        '2-40-g11-k1-005.txt',
        '2-40-g11-k1-006.txt'
    ]
    image_files_list = [os.path.join(base_path, name) for name in image_file_names]

    # Filter for existing files to avoid errors if some are missing
    image_files_list_existing = [f for f in image_files_list if os.path.exists(f)]

    if not image_files_list_existing:
        print("Error: No image files found at the specified paths. Please check.")
    else:
        print(f"Found {len(image_files_list_existing)} image files to process:")
        for f_path in image_files_list_existing:
            print(f"  - {f_path}")

    control_points_data_file = os.path.join(base_path, '40mm控制点.txt')

    if not os.path.exists(control_points_data_file):
        print(f"Error: Control point file not found: {control_points_data_file}")
    elif not image_files_list_existing:
        print("Skipping adjustment due to missing image files.")
    else:
        back_intersection(image_files_list_existing, control_points_data_file)