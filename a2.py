#!/bin/python3

import geomproc
import numpy as np
import random

PC_SUB_SAMPLES = 200 # used during ransac sampling
FULL_PC_SAMPLES = 5000 # used during point descriptor calculation
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


def get_best_n_matches(target, descriptors, n=DESC_FILTER_TARGET):
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

# returns 3 candidate pairs [source_index, target_index, distance (0)]
def get_candidate_pairs(target_desc, source_desc):
    # select three candidates and their corresponding best matches
    candidate_pairs = []
    for i in range(3):
        target_ix = random.randint(0, len(target_desc)-1)
        target = target_desc[target_ix]
        source_matches = get_best_n_matches(target, source_desc, DESC_FILTER_TARGET)
        candidate_ix = random.randint(0, source_matches.shape[0]-1)
        candidate_pairs.append([source_matches[candidate_ix], target_ix, 0])
        # candidate_pairs.append([target_ix, matches[candidate_ix], 0])

    return np.array(candidate_pairs)

def compute_errors(reference_pc, transformed_points):
    result = []
    tree = geomproc.kdtree.KDTree(reference_pc.point.tolist(), np.arange(reference_pc.point.shape[0]).tolist())
    for src_ix, src_point in enumerate(transformed_points):
        [[target_point, target_ix]] = tree.nn_query(src_point)
        dist = np.linalg.norm((target_point - src_point))
        result.append([src_ix, target_ix, dist])
    return np.array(result)

def ransac_align(target_pc, target_sampled, source_pc, source_sampled, iterations=RANSAC_ITERATIONS):
    best_rot = np.identity(3)
    best_trans = np.zeros((3, 1))
    best_inlier_count = 0
    target_desc, source_desc = compute_descriptors(target_pc, target_sampled, source_pc, source_sampled)
    for i in range(iterations):
        if i % 10 == 0:
            print(i)
        candidate_pairs = get_candidate_pairs(target_desc, source_desc)
        rot, trans = geomproc.transformation_from_correspondences(source_sampled, target_sampled, candidate_pairs)
        transf_src_points = geomproc.apply_transformation(source_sampled.point, rot, trans)
        errors = compute_errors(target_sampled, transf_src_points)
        inlier_indices = np.where(errors[:,2] < EPSILON)[0]
        if inlier_indices.shape[0] > best_inlier_count:
            source_transf_pc = target_sampled.copy()
            best_rot, best_trans = geomproc.transformation_from_correspondences(source_transf_pc, target_sampled, errors.astype(np.int_))
            best_inlier_count = inlier_indices.shape[0]
            best_rot = rot
            best_trans = trans
            print(f"BEST {best_inlier_count}")
            # recompute rot, trans for inliers

    print(best_inlier_count)
    return best_rot, best_trans


def main():
    rot, trans = random_transform()

    target_mesh = geomproc.load("../GeomProc/meshes/bunny.obj")
    target_mesh.normalize()
    source_mesh = target_mesh.copy()

    source_mesh.vertex = geomproc.apply_transformation(source_mesh.vertex, rot, trans)
    source_mesh.compute_vertex_and_face_normals()

    target_pc         = target_mesh.sample(FULL_PC_SAMPLES)
    target_sampled    = target_mesh.sample(PC_SUB_SAMPLES)
    source_pc         = source_mesh.sample(FULL_PC_SAMPLES)
    source_sampled    = source_mesh.sample(PC_SUB_SAMPLES)

    align_rot, align_trans = ransac_align(target_pc, target_sampled, source_pc, source_sampled)

    pt1 = geomproc.create_points(target_pc.point, color=[1, 0, 0])
    pt2 = geomproc.create_points(geomproc.apply_transformation(source_pc.point, align_rot, align_trans), color=[0, 1, 0])
    pt3 = geomproc.create_points(source_pc.point, color=[0, 0, 1])
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