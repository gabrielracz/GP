#!/bin/python3

import geomproc
import numpy as np
import random

NUM_SAMPLES = 100 # used during ransac sampling
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


def ransac_align(pc1_full, pc1_samples, pc2_full, pc2_samples):
    # create NUM_SAMPLES spin images to act as our ransac point pool
    # compute descriptors
    options = geomproc.spin_image_options()
    desc1 = geomproc.spin_images(pc1_samples, pc1_full, options)
    desc2 = geomproc.spin_images(pc2_samples, pc2_full, options)

    # select three candidates and their corresponding best matches
    candidate_pairs = []
    for i in range(3):
        ix = random.randint(0, desc1.shape[0])
        matches = geomproc.best_match([desc1[ix]], desc2)
        matches, _ = geomproc.filter_correspondences(corr=matches[:,:2], dist=full_matches[:, 2], keep=DESC_FILTER_TARGET/NUM_SAMPLES)


    full_matches = geomproc.best_match(desc1, desc2)
    [matches, distances] = geomproc.filter_correspondences(corr=full_matches[:,:2], dist=full_matches[:, 2], keep=DESC_FILTER_TARGET/NUM_SAMPLES)
    matches = np.hstack((matches, np.zeros((matches.shape[0], 1))))
    matches = np.array(matches, dtype=np.int_)
    print(matches)

    for match in sampled_matches:
        print(pc1_samples.normal[match[0]], pc1_samples.normal[match[1]])

    [rot, trans] = geomproc.transformation_from_correspondences(pc1_samples, pc2_samples, matches)
    # return rot, trans

    # sample the "same" 3 points from each pointcloud
    # Derive a transformation from the point match
    sample_indices = np.random.randint(matches.shape[0], size=(3,))
    sampled_matches = (matches[sample_indices[0]], matches[sample_indices[1]], matches[sample_indices[2]])
    print(sampled_matches)
    for match in sampled_matches:
        print(pc1_samples.normal[match[0]], pc1_samples.normal[match[1]])
    exit(0)

    destination_points = (pc1_samples.point[sampled_matches[0][0]], 
                          pc1_samples.point[sampled_matches[1][0]],
                          pc1_samples.point[sampled_matches[2][0]])
    source_points = (pc2_samples.point[sampled_matches[0][1]], 
                     pc2_samples.point[sampled_matches[1][1]],
                     pc2_samples.point[sampled_matches[2][1]])

    # C = AB-1
    A = create_local_basis(destination_points)
    B = create_local_basis(source_points)

    A = np.hstack(A, np.array([0, 0, ]))

    C, error = np.linalg.lstsq(A, B, rcond=None)[:2]
    return C


def main():
    rot = np.identity(3)
    # trans = np.array([0.5, 0.5, 0.5]).reshape((3, 1))
    trans = np.zeros((3, 1))

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