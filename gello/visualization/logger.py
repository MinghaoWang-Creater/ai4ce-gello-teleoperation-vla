import rerun as rr
import mujoco
from dm_control import mjcf
import numpy as np
from dm_control.mjcf.element import _AttachableElement
from trimesh.visual.texture import TextureVisuals
from PIL import Image
import trimesh
from scipy.spatial.transform import Rotation as R


def attach_gripper_to_robot(
    robot_arm: mjcf.RootElement,
    robot_gripper: mjcf.RootElement,
):
    physics = mjcf.Physics.from_mjcf_model(robot_gripper)

    attachment_site = robot_arm.find("site", "attachment_site")
    assert isinstance(attachment_site, _AttachableElement), (
        "Attachment site not found in the robot arm model."
    )

    arm_key = robot_arm.find("key", "home")
    assert isinstance(arm_key, mjcf.Element)

    gripper_key = robot_gripper.find("key", "home")
    if gripper_key is None:
        arm_key.set_attributes(
            ctrl=np.concatenate(
                [arm_key.get_attributes()["ctrl"], np.zeros(physics.model.nu)]
            ),
            qpos=np.concatenate(
                [arm_key.get_attributes()["qpos"], np.zeros(physics.model.nq)]
            ),
        )
    else:
        assert isinstance(gripper_key, mjcf.Element)
        arm_key.set_attributes(
            ctrl=np.concatenate(
                [arm_key.get_attributes()["ctrl"], gripper_key.get_attributes()["ctrl"]]
            ),
            qpos=np.concatenate(
                [arm_key.get_attributes()["qpos"], gripper_key.get_attributes()["qpos"]]
            ),
        )

    attachment_site.attach(robot_gripper)


def build_scene(
    robot_xml_path: str, gripper_xml_path: str | None = None
) -> tuple[mjcf.RootElement, dict[str, bytes]]:
    arena = mjcf.RootElement("arena")

    robot_arm = mjcf.from_path(robot_xml_path)
    robot_arm.get_assets()
    if gripper_xml_path:
        robot_gripper = mjcf.from_path(gripper_xml_path)
        attach_gripper_to_robot(robot_arm, robot_gripper)

    world_body = arena.get_children("worldbody")
    assert isinstance(world_body, _AttachableElement), (
        "World body not found in the MJCF model."
    )
    world_body.attach(robot_arm)

    assets: dict[str, bytes] = {}
    for asset in arena.asset.all_children():
        if asset.tag == "mesh":
            f = asset.file
            assets[f.get_vfs_filename()] = asset.file.contents

    return arena, assets


class RerunMJCFLogger:
    def __init__(
        self,
        xml_string: str,
        assets: dict[str, bytes],
        entity_path: str | None = None,
    ):
        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self.entity_path = entity_path
        self.joint_to_transform = {}

    def log(self):
        world_geometry_info, *links_geometry_infos = self._parse_geometries()
        (world_link_info, *link_infos), (world_joint_info, *links_joint_infos) = (
            self._parse_links()
        )

        # There should be a topological sort

        # Let's assume we've done the topological sort
        for link_idx, (link_info, link_geometry_info, link_joint_info) in enumerate(
            zip(
                link_infos,
                links_geometry_infos,
                links_joint_infos,
            )
        ):
            # build the path to the link
            path = [link_info["name"]]
            parent_idx = link_info["parent_idx"]
            while parent_idx != -1:
                # add parent's joints in reverse order first
                for parent_joint in links_joint_infos[parent_idx][::-1]:
                    path.append(parent_joint["name"])
                path.append(link_infos[parent_idx]["name"])
                parent_idx = link_infos[parent_idx]["parent_idx"]
            if self.entity_path is not None:
                path.append(self.entity_path)
            path.reverse()

            rr.log(
                path,
                # w,x,y,z -> x,y,z,w
                rr.Transform3D(
                    translation=link_info["pos"],
                    quaternion=link_info["quat"][[1, 2, 3, 0]],
                ),
            )

            # log joints
            for joint_info in link_joint_info:
                path.append(joint_info["name"])
                self.joint_to_transform["/".join(path)] = (
                    joint_info["pos"],
                    joint_info["quat"][[1, 2, 3, 0]],
                    joint_info["axis"],
                )
                rr.log(
                    path,
                    rr.Transform3D(
                        translation=joint_info["pos"],
                        quaternion=joint_info["quat"][[1, 2, 3, 0]],
                    ),
                )

            # log visuals
            visual_path = path.copy() + ["visual"]
            for geometry_info in link_geometry_info:
                if geometry_info["type"] == "mesh":
                    rr.log(
                        visual_path + [geometry_info["name"]],
                        rr.Transform3D(
                            translation=geometry_info["pos"],
                            quaternion=geometry_info["quat"][[1, 2, 3, 0]],
                        ),
                    )
                    log_trimesh(
                        visual_path + [geometry_info["name"]],
                        geometry_info["mesh"],
                    )
            rr.log(visual_path, rr.Arrows3D(origins=[[0, 0, 0]], vectors=[[0.1, 0, 0]]))

        return self.joint_to_transform

    def _parse_geometry(self, geometry_idx: int):
        geom = self._model.geom(geometry_idx)
        assert isinstance(geom, mujoco._structs._MjModelGeomViews)

        geom_size = self._model.geom_size[geometry_idx]

        visual = None
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = self._model.mesh(geom.dataid[0])
            assert isinstance(mesh, mujoco._structs._MjModelMeshViews)
            vertex_start = mesh.vertadr[0]
            vertex_num = mesh.vertnum[0]
            vertex_end = vertex_start + vertex_num

            face_start = mesh.faceadr[0]
            face_num = mesh.facenum[0]
            face_end = face_start + face_num

            vertices = self._model.mesh_vert[vertex_start:vertex_end]
            faces = self._model.mesh_face[face_start:face_end]
            face_normals = self._model.mesh_normal[face_start:face_end]
            visual = None
            vertex_colors = None

            if geom.matid >= 0:
                material = self._model.mat(geom.matid)
                assert isinstance(material, mujoco._structs._MjModelMaterialViews)
                vertex_colors = material.rgba
                assert isinstance(material, mujoco._structs._MjModelMaterialViews)
                texture_id_RGB = material.texid[mujoco.mjtTextureRole.mjTEXROLE_RGB]
                texture_id_RGBA = material.texid[mujoco.mjtTextureRole.mjTEXROLE_RGBA]
                texture_id = texture_id_RGB if texture_id_RGB >= 0 else texture_id_RGBA
                if texture_id >= 0:
                    texture = self._model.tex(texture_id)
                    assert isinstance(texture, mujoco._structs._MjModelTextureViews)
                    texture_vertex_start = int(self._model.mesh_texcoordadr[mesh.id])
                    texture_vertex_num = int(self._model.mesh_texcoordnum[mesh.id])
                    texture_vertex_end = texture_vertex_start + texture_vertex_num
                    if texture_vertex_start >= 0:
                        vertices = np.zeros((texture_vertex_num, 3))
                        faces = self._model.mesh_facetexcoord[face_start:face_end]
                        for face_id in range(face_start, face_end):
                            for i in range(3):
                                mesh_vertex_id = self._model.mesh_face[face_id, i]
                                texture_vertex_id = self._model.mesh_facetexcoord[
                                    face_id, i
                                ]
                                vertices[texture_vertex_id] = self._model.mesh_vert[
                                    mesh_vertex_id + vertex_start
                                ]

                        uv = self._model.mesh_texcoord[
                            texture_vertex_start:texture_vertex_end
                        ]
                        uv[:, 1] = 1 - uv[:, 1]

                        H, W, C = (
                            texture.height[0],
                            texture.width[0],
                            texture.nchannel[0],
                        )
                        image_array = self._model.tex_data[
                            texture.adr[0] : (texture.adr[0] + H * W * C)
                        ].reshape(H, W, C)
                        uv = uv * material.texrepeat
                        visual = TextureVisuals(
                            uv=uv,
                            image=Image.from_array(image_array),
                        )

            return {
                "type": "mesh",
                "name": mesh.name.strip("/").split("/")[-1],
                "pos": geom.pos,
                "quat": geom.quat,
                "mesh": trimesh.Trimesh(
                    vertices=vertices,
                    vertex_colors=vertex_colors,
                    faces=faces,
                    face_normals=face_normals,
                    process=False,
                    visual=visual,
                ),
                "data": geom_size,
            }

        return None

    def _parse_geometries(self):
        links_geometry_info = [[] for _ in range(self._model.nbody)]

        for geometry_idx in range(self._model.ngeom):
            if self._model.geom_bodyid[geometry_idx] < 0:
                continue

            geometry_info = self._parse_geometry(geometry_idx)
            if geometry_info is None:
                continue

            link_idx = self._model.geom_bodyid[geometry_idx]
            links_geometry_info[link_idx].append(geometry_info)

        return links_geometry_info

    def _parse_link(self, link_idx: int):
        name_start = self._model.name_bodyadr[link_idx]
        name_end = (
            self._model.name_bodyadr[link_idx + 1]
            if link_idx + 1 < self._model.nbody
            else len(self._model.names)
        )
        link_info = {
            "name": self._model.names[name_start:name_end]
            .decode("utf-8")
            .split("\x00", 1)[0]
            .strip("/")
            .split("/")[-1],
            "pos": self._model.body_pos[link_idx],
            "quat": self._model.body_quat[link_idx],
            "parent_idx": int(self._model.body_parentid[link_idx] - 1),
        }

        joint_adr = self._model.body_jntadr[link_idx]
        joint_num = self._model.body_jntnum[link_idx]

        joint_infos = []
        for joint_idx in range(joint_adr, joint_adr + max(joint_num, 1)):
            joint_info = {
                "type": self._model.jnt_type[joint_idx],
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }

            if joint_idx == -1:
                joint_info["name"] = link_info["name"]
                joint_info["axis"] = np.array([0.0, 0.0, 1.0])
                joint_info["pos"] = np.array([0.0, 0.0, 0.0])
                joint_info["n_dofs"] = 0
            else:
                joint_info["name"] = (
                    self._model.names[self._model.name_jntadr[joint_idx] :]
                    .decode("utf-8")
                    .split("\x00", 1)[0]
                    .strip("/")
                    .split("/")[-1]
                )
                joint_info["pos"] = self._model.jnt_pos[joint_idx]
                joint_info["axis"] = self._model.jnt_axis[joint_idx]
                joint_type = self._model.jnt_type[joint_idx]
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    joint_info["n_dofs"] = 1
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    joint_info["n_dofs"] = 1
                elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                    joint_info["n_dofs"] = 3
                elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
                    joint_info["n_dofs"] = 6
                else:
                    raise ValueError(
                        f"Unsupported joint type: {joint_type} for joint {joint_info['name']}"
                    )

            joint_infos.append(joint_info)

        joint_infos = [j for j in joint_infos if j["n_dofs"] > 0]

        return link_info, joint_infos

    def _parse_links(self):
        link_infos = []
        joint_infos = []

        for link_idx in range(self._model.nbody):
            l_info, j_info = self._parse_link(link_idx)

            link_infos.append(l_info)
            joint_infos.append(j_info)

        return link_infos, joint_infos


def pil_image_to_albedo_texture(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an albedo texture."""
    albedo_texture = np.asarray(image)
    if albedo_texture.ndim == 2:
        # If the texture is grayscale, we need to convert it to RGB since
        # Rerun expects a 3-channel texture.
        # See: https://github.com/rerun-io/rerun/issues/4878
        albedo_texture = np.stack([albedo_texture] * 3, axis=-1)
    return albedo_texture


def log_trimesh(entity_path: str | list[str], mesh: trimesh.Trimesh) -> None:
    vertex_colors = albedo_texture = vertex_texcoords = None

    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        vertex_colors = mesh.visual.vertex_colors
    elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        trimesh_material = mesh.visual.material

        if mesh.visual.uv is not None:
            vertex_texcoords = mesh.visual.uv
            # Trimesh uses the OpenGL convention for UV coordinates, so we need to flip the V coordinate
            # since Rerun uses the Vulkan/Metal/DX12/WebGPU convention.
            vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]

        if isinstance(trimesh_material, trimesh.visual.material.PBRMaterial):
            if trimesh_material.baseColorTexture is not None:
                albedo_texture = pil_image_to_albedo_texture(
                    trimesh_material.baseColorTexture
                )
            elif trimesh_material.baseColorFactor is not None:
                vertex_colors = trimesh_material.baseColorFactor
        elif isinstance(trimesh_material, trimesh.visual.material.SimpleMaterial):
            if trimesh_material.image is not None:
                albedo_texture = pil_image_to_albedo_texture(trimesh_material.image)
            else:
                vertex_colors = mesh.visual.to_color().vertex_colors

    rr.log(
        entity_path,
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=vertex_colors,
            albedo_texture=albedo_texture,
            vertex_texcoords=vertex_texcoords,
        ),
        static=True,
    )


if __name__ == "__main__":
    ROBOT_XML_PATH = "./third_party/mujoco_menagerie/universal_robots_ur10e/ur10e.xml"
    GRIPPER_XML_PATH = "./third_party/mujoco_menagerie/robotiq_2f85/2f85.xml"

    rr.init("dev")
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
    rr.set_time("recording_step", sequence=0)

    arena, assets = build_scene(ROBOT_XML_PATH, GRIPPER_XML_PATH)
    logger = RerunMJCFLogger(arena.to_xml_string(), assets, "robot")
    logger.log()

    import pickle
    import sys
    from natsort import natsorted
    from pathlib import Path

    sys.path.append("./src")

    from rerun_mjcf_logger.utils import log_angle_rot

    DATA_PATH = Path("./bc_data/gello/0430_213005")
    all_logs = natsorted(DATA_PATH.glob("*.pkl"))

    def read_log(log_path: Path):
        with log_path.open("rb") as f:
            log = pickle.load(f)

        return log

    for idx, log_path in enumerate(all_logs):
        rr.set_time("recording_step", sequence=idx)
        joint_position = read_log(log_path)["joint_positions"][:6]
        for i in range(6):
            log_angle_rot(logger.joint_to_transform, i, joint_position[i])