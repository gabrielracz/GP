#!/bin/python3
import numpy as np
import scipy
import scipy.spatial
import geomproc
import math

PC_SAMPLES = 4000
MAX_NEIGHBOUR_DISTANCE = 0.25
EPSILON = 0.01


def main():
    mesh = geomproc.load("../GeomProc/meshes/bunny.obj")
    mesh.normalize()
    mesh.compute_vertex_and_face_normals()
    pc = mesh.sample(PC_SAMPLES)
    tree = scipy.spatial.KDTree(pc.point)
    surf = geomproc.impsurf()

    def kernel(x):
        nearest_point, nearest_index = tree.query(x, workers=-1, k=1)
        neighbour_indices = tree.query_ball_point(pc.point[nearest_index], MAX_NEIGHBOUR_DISTANCE, workers=-1)
        denom = 0
        acc = 0
        for nb_point, nb_ix in zip(pc.point[neighbour_indices], neighbour_indices):
            diff = (x - nb_point)
            dist = np.linalg.norm(diff)
            sigma = math.exp(-dist**2 / EPSILON)
            acc += sigma * np.dot(pc.normal[nb_ix], diff)
            denom += sigma
        denom = max(denom, 0.00000001)
        result = acc / denom
        return result

    surf.evaluate = kernel

    rec = geomproc.marching_cubes(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]), 32, surf.evaluate)
    rec.save("marched.obj")
    return

if __name__ == "__main__":
    main()