#!/bin/python3

import geomproc
import numpy as np
import random

NUM_SAMPLES = 300 # used during ransac sampling
NUM_FULL_SAMPLES = 1000 # used during point descriptor calculation
DESC_FILTER_TARGET = 20
RANSAC_ITERATIONS = 50
EPSILON =0.01

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


def get_candidate_pairs(pc1_full, pc1_samples, pc2_full, pc2_samples):
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

    return candidate_pairs

def compute_errors(originalpc, transformed_points):
    result = []
    tree = geomproc.kdtree.KDTree(originalpc.point.tolist(), np.arange(originalpc.point.shape[0]).tolist())
    for i, trpoint in enumerate(transformed_points):
        [[ogpoint, ogix]] = tree.nn_query(trpoint)
        dist = np.linalg.norm((ogpoint - trpoint))
        # print(ogpoint, "   ", trpoint)
        result.append([ogix, i, dist])
    return np.array(result)

def ransac_align(pc1_full, pc1_samples, pc2_full, pc2_samples, iterations=RANSAC_ITERATIONS):
    best_rot = np.identity(3)
    best_trans = np.zeros((3, 1))
    best_inlier_count = 0
    for i in range(iterations):
        candidate_pairs = get_candidate_pairs(pc1_full, pc1_samples, pc2_full, pc2_samples)
        rot, trans = geomproc.transformation_from_correspondences(pc1_samples, pc2_samples, np.array(candidate_pairs))
        transformed_points = geomproc.apply_transformation(pc2_samples.point, rot, trans)
        errors = compute_errors(pc1_samples, transformed_points)
        inlier_indices = np.where(errors[:,2] < EPSILON)[0]
        if inlier_indices.shape[0] > best_inlier_count:
            best_rot = rot
            best_trans = trans
            best_inlier_count = inlier_indices.shape[0]
            # recompute rot, trans for inliers


    return best_rot, best_trans


def main():

    rot = np.identity(3)
    # trans = np.array([0.5, 0.5, 0.5]).reshape((3, 1))
    trans = np.zeros((3, 1))

    mesh1 = geomproc.load("../GeomProc/meshes/bunny.obj")
    mesh1.normalize()
    mesh1.compute_connectivity()
    mesh2 = mesh1.copy()
    mesh1.compute_vertex_and_face_normals()
    # mesh2.vertex = geomproc.apply_transformation(mesh1.vertex, rot, trans)
    mesh2.compute_vertex_and_face_normals()

    pc1 = mesh1.sample(NUM_FULL_SAMPLES)
    pc2 = mesh2.sample(NUM_FULL_SAMPLES)
    pc1_samples = mesh1.sample(NUM_SAMPLES)
    pc2_samples = mesh2.sample(NUM_SAMPLES)

    align_rot, align_trans = ransac_align(pc1, pc1_samples, pc2, pc2_samples)
    # align_rot, align_trans = ransac_align(pc1, pc1_samples, pc1.copy(), pc1_samples.copy())

    # pt1 = geomproc.create_points(orig, color=[1, 0, 0])
    # pt2 = geomproc.create_points(source, color=[0, 1, 0])
    # # Combine everything together
    # result = geomproc.mesh()
    # result.append(pt1)
    # result.append(pt2)
    # result.append(pt3)
    # # Save the mesh
    # wo = geomproc.write_options()
    # wo.write_vertex_colors = True
    # result.save('output.obj', wo)
    print("RANSAC")
    print(align_rot, align_trans)

if __name__ == "__main__":
    main()