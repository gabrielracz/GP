#!/bin/python3

import geomproc
import numpy as np
import random

NUM_SAMPLES = 200 # used during ransac sampling
NUM_FULL_SAMPLES = 1000 # used during point descriptor calculation
DESC_FILTER_TARGET = 20

def normalize(v):
    norm = np.linalg.norm(v)
    return v / np.linalg.norm(v) if norm > 0 else v

def random_rotation():
    rnd = np.random.random((3, 3))
    [q, r] = np.linalg.qr(rnd)
    orig_rot = q
    orig_trans = np.random.random((3, 1))

def create_local_basis(points):
    (p1, p2, p3) = points
    v1 = normalize(p2 - p1)
    v2 = normalize(p3 - p1)
    basis1 = v1
    norm = normalize(np.cross(v1, v2))
    basis2 = normalize(np.cross(basis1, norm))
    transform = np.column_stack((basis1, norm, basis2))
    return transform

def get_best_n_matches(target, descriptors, n=DESC_FILTER_TARGET):
    # descriptors is shape (n, NUM_BINS)
    # target is (1, NUM_BINS)
    # find top n results
    target = target[:, None].T # make row vec
    
    diff = descriptors - target
    error = np.linalg.norm(diff, axis=1)
    return np.argpartition(error, n)[:n] # return UNSORTED n smallest error indices

def get_transformation(A, B):
    # Compute the centroids
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # Center the points by subtracting the centroids
    A_centered = A - centroid_A[:, np.newaxis]
    B_centered = B - centroid_B[:, np.newaxis]

    # Compute the covariance matrix H
    H = B_centered @ A_centered.T

    # Perform SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure a proper rotation matrix by correcting for reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = centroid_A - R @ centroid_B
    return R, t


def 

def ransac_align(pc1_full, pc1_samples, pc2_full, pc2_samples):
    # create NUM_SAMPLES spin images to act as our ransac point pool
    # compute descriptors
    options = geomproc.spin_image_options()
    desc1 = geomproc.spin_images(pc1_samples, pc1_full, options)
    desc2 = geomproc.spin_images(pc2_samples, pc2_full, options)

    # select three candidates and their corresponding best matches
    candidate_pairs = []
    for i in range(3):
        target_ix = random.randint(0, len(desc1)-1)
        target = desc1[target_ix]
        matches = get_best_n_matches(target, desc2, DESC_FILTER_TARGET)
        candidate_ix = random.randint(0, matches.shape[0]-1)
        candidate_pairs.append([target_ix, matches[candidate_ix], 0])

    rot, trans = geomproc.transformation_from_correspondences(pc1_samples, pc2_samples, np.array(candidate_pairs))

    # orig = np.zeros((3,3))
    # source = np.zeros((3,3))
    # transformed = np.zeros((3,3))
    # for i in range(3):
    #     orig[i] = np.array(pc1_samples.point[candidate_pairs[i][0]])
    #     src = pc2_samples.point[candidate_pairs[i][1]]
    #     source[i] = np.array(src)
    #     transformed[i] =  np.array(((rot @ src).reshape(3, 1) + trans).reshape(1, 3))
    #     # print((rot @ src).reshape(3, 1) + trans)

    # print(orig)
    # print(source)
    # print(transformed)
    # # print(trans)
    # # print(source)

    # pt1 = geomproc.create_points(orig, color=[1, 0, 0])
    # pt2 = geomproc.create_points(source, color=[0, 1, 0])
    # pt3 = geomproc.create_points(transformed, color=[0, 0, 1])
    # # Combine everything together
    # result = geomproc.mesh()
    # result.append(pt1)
    # result.append(pt2)
    # result.append(pt3)
    # # Save the mesh
    # wo = geomproc.write_options()
    # wo.write_vertex_colors = True
    # result.save('output.obj', wo)
    return rot, trans


def main():

    rot = np.identity(3)
    trans = np.array([0.5, 0.5, 0.5]).reshape((3, 1))
    # trans = np.zeros((3, 1))

    mesh1 = geomproc.load("../GeomProc/meshes/bunny.obj")
    mesh1.normalize()
    mesh1.compute_connectivity()
    mesh2 = mesh1.copy()
    mesh1.compute_vertex_and_face_normals()
    mesh2.vertex = geomproc.apply_transformation(mesh1.vertex, rot, trans)
    mesh2.compute_vertex_and_face_normals()

    pc1 = mesh1.sample(NUM_FULL_SAMPLES)
    pc2 = mesh2.sample(NUM_FULL_SAMPLES)
    pc1_samples = mesh1.sample(NUM_SAMPLES)
    pc2_samples = mesh2.sample(NUM_SAMPLES)

    align_rot, align_trans = ransac_align(pc1, pc1_samples, pc2, pc2_samples)
    print("RANSAC")
    print(align_rot, align_trans)

if __name__ == "__main__":
    main()