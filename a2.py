#!/bin/python3

import geomproc
import numpy as np
import random

NUM_SAMPLES = 200 # used during ransac sampling
NUM_FULL_SAMPLES = 5000 # used during point descriptor calculation
DESC_FILTER_TARGET = 10
RANSAC_ITERATIONS = 1000
EPSILON =0.1

def save_points(p1, p2=[], p3=[]):
    pt1 = geomproc.create_points(p1, color=[1, 0, 0])
    pt2 = geomproc.create_points(p2, color=[0, 1, 0])
    pt3 = geomproc.create_points(p3, color=[0, 0, 1])
    # Combine everything together
    result = geomproc.mesh()
    result.append(pt1)
    result.append(pt2)
    result.append(pt3)
    # Save the mesh
    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    result.save('output.obj', wo)

def normalize(v):
    norm = np.linalg.norm(v)
    return v / np.linalg.norm(v) if norm > 0 else v

def random_transform():
    rnd = np.random.random((3, 3))
    [q, r] = np.linalg.qr(rnd)
    return q, np.random.random((3, 1))

def create_local_basis(points):
    (p1, p2, p3) = points
    v1 = normalize(p2 - p1)
    v2 = normalize(p3 - p1)
    basis1 = v1
    norm = normalize(np.cross(v1, v2))
    basis2 = normalize(np.cross(basis1, norm))
    transform = np.column_stack((basis1, norm, basis2))
    return transform

def estimate_transformation(A, B):
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
    # print(R, t)

    return R, t[:, None]


def get_best_n_matches(target, descriptors, n=DESC_FILTER_TARGET, target_ix=0):
    target = target[:, None].T # make row vec
    diff = descriptors - target
    error = np.linalg.norm(diff, axis=1)
    # lowest_n_indices = np.argpartition(error, n)[:n] # return UNSORTED n smallest error indices
    sorted_indices = np.argsort(error)
    return sorted_indices[:n]

def compute_descriptors(pc1_full, pc1_samples, pc2_full, pc2_samples):
    options = geomproc.spin_image_options()
    options.height_bins = 10
    options.radial_bins = 50
    desc1 = geomproc.spin_images(pc1_samples, pc1_full, options)
    desc2 = geomproc.spin_images(pc2_samples, pc2_full, options)
    return desc1, desc2

def get_candidate_pairs(desc1, desc2):
    # select three candidates and their corresponding best matches
    candidate_pairs = []
    for i in range(3):
        target_ix = random.randint(0, len(desc1)-1)
        target = desc1[target_ix]
        # print(target_ix)
        matches = get_best_n_matches(target, desc2, DESC_FILTER_TARGET, target_ix)
        candidate_ix = random.randint(0, matches.shape[0]-1)
        candidate_pairs.append([target_ix, matches[candidate_ix], 0])
        # print(np.linalg.norm(desc1[target_ix] - desc2[matches[candidate_ix]]))
    # exit(0)
    return np.array(candidate_pairs)

def compute_errors(originalpc, transformed_points):
    result = []
    tree = geomproc.kdtree.KDTree(originalpc.point.tolist(), np.arange(originalpc.point.shape[0]).tolist())
    for i, trpoint in enumerate(transformed_points):
        [[ogpoint, ogix]] = tree.nn_query(trpoint)
        dist = np.linalg.norm((ogpoint - trpoint))
        result.append([ogix, i, dist])
    return np.array(result)

def ransac_align(pc1_full, pc1_samples, pc2_full, pc2_samples, iterations=RANSAC_ITERATIONS):
    best_rot = np.identity(3)
    best_trans = np.zeros((3, 1))
    best_inlier_count = 0
    desc1, desc2 = compute_descriptors(pc1_full, pc1_samples, pc2_full, pc2_samples)
    for i in range(iterations):
        if i % 10 == 0:
            print(i)
        candidate_pairs = get_candidate_pairs(desc1, desc2)
        rot, trans = geomproc.transformation_from_correspondences(pc1_samples, pc2_samples, candidate_pairs)
        transformed_points = geomproc.apply_transformation(pc2_samples.point, rot, trans)
        errors = compute_errors(pc1_samples, transformed_points)
        inlier_indices = np.where(errors[:,2] < EPSILON)[0]
        if inlier_indices.shape[0] > best_inlier_count:
            pctransf = pc2_samples.copy()
            # pctransf.point = geomproc.apply_transformation(pc2_samples.point, rot, trans)
            best_rot, best_trans = geomproc.transformation_from_correspondences(pc1_samples, pctransf, errors.astype(np.int_))
            best_inlier_count = inlier_indices.shape[0]
            best_rot = rot
            best_trans = trans
            print("BEST")
            # recompute rot, trans for inliers

    print(best_inlier_count)
    return best_rot, best_trans


def main():

    # rot = np.identity(3)
    rot, trans = random_transform()
    # rot = np.identity(3)
    # trans = np.array([5, 0, 0])[:, None]

    mesh1 = geomproc.load("../GeomProc/meshes/bunny.obj")
    mesh1.normalize()
    # mesh1.compute_connectivity()
    # mesh1.compute_vertex_and_face_normals()
    mesh2 = mesh1.copy()

    mesh2.vertex = geomproc.apply_transformation(mesh2.vertex, rot, trans)
    mesh2.compute_connectivity()
    mesh2.compute_vertex_and_face_normals()


    pc1         = mesh1.sample(NUM_FULL_SAMPLES)
    pc1_samples = mesh1.sample(NUM_SAMPLES)
    pc2         = mesh2.sample(NUM_FULL_SAMPLES)
    pc2_samples = mesh2.sample(NUM_SAMPLES)
    # pc2 = pc1.copy()
    # pc2_samples = pc1_samples.copy()


    align_rot, align_trans = ransac_align(pc1, pc1_samples, pc2, pc2_samples)
    # align_rot, align_trans = ransac_align(pc1, pc1_samples, mesh1.sample(NUM_FULL_SAMPLES), mesh1.sample(NUM_SAMPLES))
    # align_rot, align_trans = ransac_align(pc1, pc1_samples, pc1.copy(), pc1_samples.copy())

    pt1 = geomproc.create_points(pc1.point, color=[1, 0, 0])
    pt2 = geomproc.create_points(geomproc.apply_transformation(pc2.point, align_rot, align_trans), color=[0, 1, 0])
    pt3 = geomproc.create_points(pc2.point, color=[0, 0, 1])
    # Combine everything together
    result = geomproc.mesh()
    result.append(pt1)
    result.append(pt2)
    result.append(pt3)
    # Save the mesh
    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    result.save('output.obj', wo)
    print("RANSAC")
    print("OG:")
    print(rot, trans)
    print("SOLVED:")
    print(align_rot, "\n", align_trans)

if __name__ == "__main__":
    main()