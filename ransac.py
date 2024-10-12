#!/bin/python3

import geomproc
import numpy as np
import random

PC_SUB_SAMPLES = 200 
FULL_PC_SAMPLES = 5000 
DESC_FILTER_TARGET = 10
RANSAC_ITERATIONS = 1000
INLIER_ERROR_THRESHOLD =0.1

def save_points(p1, p2=[], p3=[], filename=""):
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
    result.save(filename, wo)

def random_transform():
    rnd = np.random.random((3, 3))
    [q, r] = np.linalg.qr(rnd)
    return q, np.random.random((3, 1)) * 5

def get_best_n_matches(target, descriptors, n=DESC_FILTER_TARGET):
    target = target[:, None].T # make row vec
    diff = descriptors - target
    error = np.linalg.norm(diff, axis=1)
    lowest_n_indices = np.argpartition(error, n)[:n] # return UNSORTED n smallest error indices
    # sorted_indices = np.argsort(error)
    return lowest_n_indices

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

def ransac_align(target_pc, target_sampled, source_pc, source_sampled, iterations):
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
        inlier_indices = np.where(errors[:,2] < INLIER_ERROR_THRESHOLD)[0]
        if inlier_indices.shape[0] > best_inlier_count:
            source_transf_pc = source_sampled.copy()
            best_rot, best_trans = geomproc.transformation_from_correspondences(source_transf_pc, target_sampled, errors.astype(np.int_))
            best_inlier_count = inlier_indices.shape[0]
            print("FOUND", best_inlier_count)

    print("INLIERS:", best_inlier_count)
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

    save_points(target_pc.point,
                geomproc.apply_transformation(source_pc.point, align_rot, align_trans),
                source_pc.point,
                "output.obj")

    print("RANSAC")
    print("TARGET TRANSFORM:")
    print("rotation:")
    print(rot)
    print("translation:")
    print(trans)
    print()
    print("SOLVED TRANSFORM:")
    print("rotation:")
    print(align_rot)
    print("translation:")
    print(align_trans)

if __name__ == "__main__":
    main()