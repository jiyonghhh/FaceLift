# Copyright 2025 Adobe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Post-process a FaceLift Gaussian Splatting PLY into a vertex-colored triangle mesh.

Pipeline: load gaussians.ply -> activate + filter -> sample points per Gaussian
-> estimate normals -> screened Poisson reconstruction -> density crop +
largest-component cleanup -> bake SH DC color onto vertices -> save mesh.

CPU-only. Requires: numpy, plyfile, open3d, scipy. Install with:
    pip install open3d==0.18.0 scipy==1.11.4
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from plyfile import PlyData

try:
    import open3d as o3d
except ImportError as e:
    sys.exit(
        "open3d is required for mesh extraction. Install with:\n"
        "    pip install open3d==0.18.0\n"
        f"(import error: {e})"
    )

try:
    from scipy.spatial import cKDTree
except ImportError as e:
    sys.exit(
        "scipy is required for color baking. Install with:\n"
        "    pip install scipy==1.11.4\n"
        f"(import error: {e})"
    )


# Spherical-harmonics DC coefficient: matches gslrm/model/gaussians_renderer.py:138
SH_C0 = 0.28209479177387814


def load_gaussian_ply(path: Path) -> dict:
    """Read a FaceLift gaussians.ply and return raw fields as numpy arrays."""
    plydata = PlyData.read(str(path))
    el = plydata.elements[0]
    names = {p.name for p in el.properties}

    required = {
        "x", "y", "z",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2",
    }
    missing = required - names
    if missing:
        raise ValueError(
            f"PLY at {path} is missing expected Gaussian fields: {sorted(missing)}"
        )

    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1)
    opacity = np.asarray(el["opacity"]).astype(np.float32)
    scale = np.stack([np.asarray(el[f"scale_{i}"]) for i in range(3)], axis=1)
    # quaternion stored as (w, x, y, z) — see gslrm/model/gaussians_renderer.py:108-111
    rot = np.stack([np.asarray(el[f"rot_{i}"]) for i in range(4)], axis=1)
    f_dc = np.stack([np.asarray(el[f"f_dc_{i}"]) for i in range(3)], axis=1)

    return {
        "xyz": xyz.astype(np.float32),
        "opacity_raw": opacity.astype(np.float32),
        "scale_raw": scale.astype(np.float32),
        "rot_raw": rot.astype(np.float32),
        "f_dc": f_dc.astype(np.float32),
    }


def activate(g: dict) -> dict:
    """Apply the same activations the GSLRM model uses on raw PLY values."""
    opacity = 1.0 / (1.0 + np.exp(-g["opacity_raw"]))
    scale = np.exp(g["scale_raw"])
    rot_norm = np.linalg.norm(g["rot_raw"], axis=1, keepdims=True)
    rot_norm = np.where(rot_norm < 1e-8, 1.0, rot_norm)
    rot = g["rot_raw"] / rot_norm
    color = np.clip(SH_C0 * g["f_dc"] + 0.5, 0.0, 1.0)
    return {
        "xyz": g["xyz"],
        "opacity": opacity,
        "scale": scale,
        "rot": rot,
        "color": color,
    }


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) quaternions in (w, x, y, z) order -> (N, 3, 3) rotation matrices.

    Mirrors build_rotation in gslrm/model/gaussians_renderer.py:99-122.
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def filter_gaussians(
    g: dict,
    opacity_thres: float,
    scaling_thres: float,
    bbox: Tuple[float, float, float, float, float, float],
) -> dict:
    """Drop low-opacity / overscaled / out-of-bbox Gaussians."""
    xyz = g["xyz"]
    keep = g["opacity"] > opacity_thres
    keep &= g["scale"].max(axis=1) < scaling_thres
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    keep &= (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max)
    keep &= (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max)
    keep &= (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    return {k: v[keep] for k, v in g.items()}


def sample_points(g: dict, k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample k points per Gaussian from N(mu, R diag(s)^2 R^T). Sample 0 = the mean."""
    n = g["xyz"].shape[0]
    if n == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    R = quat_to_rotmat(g["rot"])  # (N, 3, 3)
    s = g["scale"]  # (N, 3)

    means = g["xyz"]
    colors = g["color"]

    if k <= 1:
        return means.astype(np.float32), colors.astype(np.float32)

    # Sample (k-1) offsets per Gaussian; first slot is the mean itself.
    eps = rng.standard_normal((n, k - 1, 3)).astype(np.float32)
    # offset = R @ diag(s) @ eps  ->  einsum('nij,nkj->nki', R, s[:, None, :] * eps)
    scaled = eps * s[:, None, :]  # (N, k-1, 3)
    offsets = np.einsum("nij,nkj->nki", R, scaled)  # (N, k-1, 3)

    pts = np.concatenate([means[:, None, :], means[:, None, :] + offsets], axis=1)
    pts = pts.reshape(-1, 3)
    cols = np.broadcast_to(colors[:, None, :], (n, k, 3)).reshape(-1, 3)
    return pts.astype(np.float32), cols.astype(np.float32)


def build_pointcloud(points: np.ndarray, colors: np.ndarray) -> "o3d.geometry.PointCloud":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    # Orient normals outward from the head centroid: first point them toward the
    # centroid, then flip the whole field so they point outward (the head shell
    # is roughly star-shaped around its centroid).
    centroid = points.mean(axis=0)
    pcd.orient_normals_towards_camera_location(camera_location=centroid)
    pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))
    return pcd


def poisson_mesh(
    pcd: "o3d.geometry.PointCloud", depth: int, density_quantile: float
) -> "o3d.geometry.TriangleMesh":
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False
    )
    densities = np.asarray(densities)
    if densities.size == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(
            "Poisson reconstruction produced an empty mesh. "
            "Try lowering --opacity-thres or --scaling-thres, or raise --samples-per-gaussian."
        )

    if density_quantile > 0:
        thr = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < thr)

    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Keep only the largest connected component; Poisson sometimes adds floaters.
    if len(mesh.triangles) > 0:
        clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
        cluster_n_tris = np.asarray(cluster_n_tris)
        if cluster_n_tris.size > 1:
            largest = int(np.argmax(cluster_n_tris))
            mesh.remove_triangles_by_mask(np.asarray(clusters) != largest)
            mesh.remove_unreferenced_vertices()

    return mesh


def bake_vertex_colors(
    mesh: "o3d.geometry.TriangleMesh", points: np.ndarray, colors: np.ndarray
) -> None:
    verts = np.asarray(mesh.vertices)
    if verts.shape[0] == 0:
        return
    tree = cKDTree(points)
    _, idx = tree.query(verts, k=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[idx].astype(np.float64))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a FaceLift gaussians.ply into a vertex-colored triangle mesh."
    )
    p.add_argument("input", type=str, help="Path to gaussians.ply")
    p.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output mesh path (default: <input_dir>/mesh.ply)",
    )
    p.add_argument("--opacity-thres", type=float, default=0.1)
    p.add_argument("--scaling-thres", type=float, default=0.05)
    p.add_argument(
        "--bbox", type=float, nargs=6,
        default=[-0.91, 0.91, -0.91, 0.91, -1.0, 1.0],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    p.add_argument("--samples-per-gaussian", type=int, default=8)
    p.add_argument("--poisson-depth", type=int, default=10)
    p.add_argument("--density-quantile", type=float, default=0.01)
    p.add_argument("--target-triangles", type=int, default=0,
                   help="If > 0, decimate the final mesh to roughly this many triangles.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).expanduser()
    if not in_path.is_file():
        sys.exit(f"Input PLY not found: {in_path}")

    out_path = Path(args.output).expanduser() if args.output else in_path.with_name("mesh.ply")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading {in_path}")
    raw = load_gaussian_ply(in_path)
    n_in = raw["xyz"].shape[0]
    if n_in == 0:
        sys.exit("Input PLY contains zero Gaussians.")
    print(f"      {n_in} Gaussians loaded")

    print("[2/7] Activating + filtering")
    g = activate(raw)
    g = filter_gaussians(g, args.opacity_thres, args.scaling_thres, tuple(args.bbox))
    n_kept = g["xyz"].shape[0]
    print(f"      {n_kept} kept (opacity>{args.opacity_thres}, "
          f"scale_max<{args.scaling_thres}, in bbox)")
    if n_kept < 1000:
        sys.exit(
            f"Only {n_kept} Gaussians survived filtering — that is too few for Poisson "
            f"reconstruction. Loosen --opacity-thres / --scaling-thres / --bbox."
        )

    print(f"[3/7] Sampling points (k={args.samples_per_gaussian} per Gaussian)")
    rng = np.random.default_rng(args.seed)
    points, colors = sample_points(g, args.samples_per_gaussian, rng)
    print(f"      {points.shape[0]} points sampled")

    print("[4/7] Estimating + orienting normals")
    pcd = build_pointcloud(points, colors)

    print(f"[5/7] Poisson reconstruction (depth={args.poisson_depth})")
    mesh = poisson_mesh(pcd, args.poisson_depth, args.density_quantile)
    print(f"      raw: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    if args.target_triangles > 0 and len(mesh.triangles) > args.target_triangles:
        print(f"[6/7] Decimating to ~{args.target_triangles} triangles")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_triangles)
        mesh.remove_unreferenced_vertices()
    else:
        print("[6/7] Skipping decimation")

    print("[7/7] Baking vertex colors")
    bake_vertex_colors(mesh, points, colors)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(
        str(out_path), mesh, write_vertex_colors=True, write_vertex_normals=True
    )
    print(
        f"\nWrote {out_path}\n"
        f"  vertices : {len(mesh.vertices)}\n"
        f"  triangles: {len(mesh.triangles)}"
    )


if __name__ == "__main__":
    main()