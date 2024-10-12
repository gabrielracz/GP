import geomproc
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

MEAN_CURV = 2
GAUSSIAN_CURV = 3

def create_local_transform(normal):
    normal = normal / np.linalg.norm(normal)
    # chose arbitrary vector that isn't inline with normal
    arbitrary = np.array([1, 0, 0]) if not abs(normal[0]) > 0.99 else np.array([0, 1, 0])
    x_basis = np.cross(normal, arbitrary)
    y_basis= np.cross(normal, x_basis)
    local_transform = np.column_stack((x_basis / np.linalg.norm(x_basis),
                                       y_basis / np.linalg.norm(y_basis),
                                       normal))
    return local_transform

def get_neighbours(mesh, vert_id, ring):
    neighbour_set = {vert_id}
    last_added = [vert_id]
    for i in range(ring):
        to_search = last_added.copy()
        last_added.clear()
        for j in to_search:
            for n in mesh.viv[j]:
                if n not in neighbour_set:
                    last_added.append(n)
                neighbour_set.add(n)
    return list(neighbour_set)


def add_vertex_sample(A, b, index, local_transform, center_vertex, neighbour):
    v = np.dot((neighbour - center_vertex), local_transform)
    row = np.array((v[0]**2, v[1]**2, v[0]*v[1], v[0], v[1], 1.0))
    A[index] = row
    b[index] = v[2]

def fit_quadratic_surface(mesh, vert_id, ring=1):
    center_vertex = mesh.vertex[vert_id]

    normal = mesh.vnormal[vert_id]
    local_transform = create_local_transform(normal)
    neighbours = get_neighbours(mesh, vert_id, ring)
    A = np.zeros((len(neighbours)+1, 6))
    A[0] = np.array([0., 0., 0., 0., 0., 1.])
    z = np.zeros((len(neighbours)+1))
    for i in range(len(neighbours)):
        nb = mesh.vertex[neighbours[i]]
        add_vertex_sample(A, z, i+1, local_transform, center_vertex, nb)

    coefficients = np.linalg.lstsq(A, z, rcond=1)[0]
    return coefficients

def solve_curvature(polynomial_surface_coefficients, method):
    a, b, c, d, e, f = polynomial_surface_coefficients

    # df/dx = 2ax + cy + d
    fx = d
    # d^2f/dx^2 = 2a
    fxx = 2*a

    # df/dy = 2by + cx + e
    fy = e
    # d^2f/dy^2 = 2b
    fyy = 2*b

    #d^2f/dxdy = c  (first dy, then dx)
    fxy = c

    E = 1 + fx**2
    F = fx * fy
    G = 1 + fy**2
    norm = math.sqrt(1 + fx**2 + fy**2)
    L = fxx/norm
    M = fxy/norm
    N = fyy/norm

    gaussian_curvature = (L*N - M**2)/(E*G - F**2)
    mean_curvature = -(E*N + G*L - 2*F*M)/(2*(E*G - F**2))

    return gaussian_curvature if method == GAUSSIAN_CURV else mean_curvature

def compute_curvature(mesh, method=GAUSSIAN_CURV, ring=1):
    curvatures = np.zeros(mesh.vertex.shape[0])
    for vert_id in range(mesh.vertex.shape[0]):
        surface = fit_quadratic_surface(mesh, vert_id, ring)
        curvatures[vert_id] = solve_curvature(surface, method)
    return curvatures

def filter_extreme_values(data, percentile):
    if len(data) == 0:
        return data
    lower_percentile = 100 - percentile
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def plot_curvatures(curvatures_ring1, geomproc_curvature, num_bins, title=''):
    curvatures_filtered1 = filter_extreme_values(curvatures_ring1, 99)
    geomproc_filtered = filter_extreme_values(geomproc_curvature, 99)

    bin_edges = np.histogram_bin_edges([curvatures_filtered1, geomproc_filtered], bins=num_bins)
    plt.hist(curvatures_filtered1, bins=bin_edges,
             histtype='stepfilled', alpha=0.5, label='polynomial')
    plt.hist(geomproc_filtered, bins=bin_edges,
             histtype='stepfilled', alpha=0.5, label='analytic')
    plt.xlabel('Curvature Value')
    plt.ylabel('Occurrences')
    plt.legend()
    plt.title(title)   

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 ./curv.py [input mesh] [g or m]")
        exit(-1)

    mesh = geomproc.load(sys.argv[1])
    mesh = geomproc.create_torus(0.6, 0.4, 30, 100)
    wo = geomproc.write_options()
    wo.write_vertex_colors = True
    method = GAUSSIAN_CURV if sys.argv[2] == 'g' else MEAN_CURV
    method_str = "Gaussian" if method == GAUSSIAN_CURV else "Mean"
    mesh.data_to_color_with_zero(mesh.curv[:, method], False)
    analytic = mesh.curv[:, method]
    mesh.save(f"Analytic-{method_str}.obj", wo)


    mesh.normalize()
    mesh.compute_connectivity()
    mesh.compute_vertex_and_face_normals()
    mesh.compute_curvature()

    curvatures_ring1 = compute_curvature(mesh, method, ring=1)
    curvatures_ring2 = compute_curvature(mesh, method, ring=2)

    mesh.data_to_color_with_zero(curvatures_ring1, False)
    mesh.save(f"{method_str}_1-ring.obj", wo)
    mesh.data_to_color_with_zero(curvatures_ring2, False)
    mesh.save(f"{method_str}_2-ring.obj", wo)

    plot_curvatures(curvatures_ring1, analytic, 60, 
                    f"{method_str} 1-ring polynomial vs. Analytical")
    plt.show()
    plot_curvatures(curvatures_ring2, analytic, 60, 
                    f"{method_str} 2-ring polynomial vs. Analytical")
    plt.show()
    plot_curvatures(mesh.curv[:, method], analytic, 60, 
                    f"{method_str} GeomProc vs. Analytical")
    plt.show()
 
    mesh.data_to_color_with_zero(mesh.curv[:, method], False)
    mesh.save(f"{method_str}.obj", wo)

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()