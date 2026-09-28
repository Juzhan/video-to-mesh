# Third-party notices

The MIT license in this repository applies to its own code and authored resources. Dependencies, external applications and downloaded model weights retain their upstream licenses and notices; they are not relicensed by this project or bundled into the source ZIP. The following records the primary licenses declared by the locally tested packages, not every transitive component of their binary distributions.

| Component | Declared license / source |
|---|---|
| COLMAP / PyCOLMAP | BSD-3-Clause; [upstream](https://github.com/colmap/colmap) |
| SAM 2 | Apache-2.0; [upstream](https://github.com/facebookresearch/sam2), setup pins commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Open3D | MIT; [upstream](https://github.com/isl-org/Open3D) |
| OpenCV | Apache-2.0; [upstream](https://github.com/opencv/opencv), wheels include additional third-party notices |
| trimesh | MIT; [upstream](https://github.com/mikedh/trimesh) |
| CoACD | MIT; [upstream](https://github.com/SarahWeiii/CoACD) |
| PyBullet | zlib; [upstream](https://github.com/bulletphysics/bullet3) |
| PyTorch | BSD-3-Clause plus bundled third-party notices; [upstream](https://github.com/pytorch/pytorch) |
| NumPy / SciPy / scikit-image | primarily BSD-3-Clause, with component-specific notices; consult installed distribution licenses |
| plyfile | **GPL-3.0-or-later**; [upstream](https://github.com/dranjan/python-plyfile), used for point-cloud PLY I/O |
| Blender | External application, GPL; [license information](https://www.blender.org/about/license/) |
| Hunyuan3D-2 / 2mv | Optional external code and weights; Tencent Hunyuan 3D 2.0 Community License and component notices; [upstream license](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE). Not included in the source ZIP. |

Setup downloads SAM weights directly from the upstream publisher and validates their checksum; consult the upstream model/license documentation. CUDA runtime components retain NVIDIA's terms. Optional Meshy use is a separate hosted service, with its own service/data terms and charges.

When redistributing an environment or binary bundle, preserve the actual licenses and notices of the versions included; this source-only project's MIT license does not replace them. `requirements-tested.txt` records the development environment versions for reference.
