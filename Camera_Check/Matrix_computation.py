from math import sin, cos, atan, asin, pi,fabs,sqrt
import numpy as np

def compute_rotation_matrix(phi: float, omega: float, kappa: float):
    """计算旋转矩阵"""
    a1 = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa)
    a2 = -cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa)
    a3 = -sin(phi) * cos(omega)
    b1 = cos(omega) * sin(kappa)
    b2 = cos(omega) * cos(kappa)
    b3 = -sin(omega)
    c1 = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa)
    c2 = -sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa)
    c3 = cos(phi) * cos(omega)
    return np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]]),a1,a2,a3,b1,b2,b3,c1,c2,c3

def transpose_matrix(A):
    """矩阵转置"""
    return np.transpose(A)

def multiply_matrix(A, B):
    """矩阵乘法"""
    return np.dot(A, B)

def add_matrix(A, B):
    """矩阵加法"""
    return np.add(A, B)

def inverse_matrix(A):
    """矩阵求逆"""
    return np.linalg.inv(A)