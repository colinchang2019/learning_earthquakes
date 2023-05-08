# -*- encoding: utf-8 -*-
"""
@ Author  ：Chesley (chelsey@zju.edu.cn)
@ File    : rotate.py
@ Time    ：2021/11/3 0:37
"""
import numpy as np
import lie_learn.spaces.S2 as S2

from config import config


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))  # 默认数据范围在[0. 1)

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )  # 计算旋转轴，即旋转轴由phi, z两个参数控制

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def get_planar_grid(grid):
    """
    :param grid: mesh of rotated picture
    :return (theta, phi): axis in x_y planar
    """
    x, y, z = grid
    # 检查x,y,z范围
    x = np.where(x >= -1.0, x, -1.0)
    x = np.where(x <= 1.0, x, 1.0)
    y = np.where(y >= -1.0, y, -1.0)
    y = np.where(y <= 1.0, y, 1.0)
    z = np.where(z >= -1.0, z, -1.0)
    z = np.where(z <= 1.0, z, 1.0)

    # 转换为角度
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    phi = np.where(phi>=0, abs(phi), 2*np.pi+phi)

    # 转换为坐标
    n = len(x)
    # print(n)
    theta = theta*n/np.pi
    phi = phi*n/(2*np.pi)

    # 处理边界
    theta, phi = ((theta+0.5).astype(int), (phi+0.5).astype(int))
    theta = np.where(theta<n, theta, 0)
    phi = np.where(phi<n, phi, 0)
    return theta, phi

def rotate_sequence(x, randnums=None):
    b = config.b
    grid = get_projection_grid(b=b)
    rot = rand_rotation_matrix(deflection=1.0, randnums=randnums)
    # print(rot)
    rotated_grid = rotate_grid(rot, grid)
    theta, phi = get_planar_grid(rotated_grid)

    type = "float16" if config.half else "float32"
    sample = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=type)
    if theta.min() < 0 or theta.max()>=config.width or phi.min()<0 or phi.max()>=config.height:
        print(theta.min(), theta.max(), phi.min(), phi.max())
    sample[:, :, :] = x[:, theta[:, :], phi[:, :]]
    return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    """
    with gzip.open("mnist.gz", 'rb') as f:
        dataset = pickle.load(f)
    x = dataset["train"]
    """

    path = "C:/Users/chesley/Pictures/pi4/"

    x = np.zeros((2, 960, 960), dtype=np.float32)
    x1 = x.reshape((2, config.height, config.width))
    print(x1.shape)
    x1 = rotate_sequence(x1, (0.5, 0.5, 0.0))
    x1 = x1.reshape((2, config.height, config.width))
    print(x1.shape)
    """
    sample[:, :, :] = x[:, theta[:, :], phi[:, :]]
    # plt.imshow(x[-1])
    plt.imshow(sample[-1])
    plt.show()
    """