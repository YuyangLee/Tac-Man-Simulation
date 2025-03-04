import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--idx", default="46466", type=int)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--capture", default=False,action="store_true")
parser.add_argument("--obj_scale", default="0.5", type=float)
parser.add_argument("--x_offset", default="0.75", type=float)
parser.add_argument("--y_offset", default="0.0", type=float)
parser.add_argument("--z_offset", default="0.0", type=float)
args = parser.parse_args()

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

import csv
import json
import os
from collections import defaultdict
from PIL import Image
from datetime import datetime

import bezier
import carb
import matplotlib.pyplot as plt
import numpy as np
import omni.graph.action
import omni.kit
import seaborn as sns
import torch
import trimesh as tm
import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.motion_generation import (ArticulationKinematicsSolver,
                                          ArticulationMotionPolicy,
                                          LulaKinematicsSolver, RmpFlow,
                                          interface_config_loader)
from omni.isaac.urdf import _urdf
from omni.physx.scripts import physicsUtils
from omni.physx.scripts.utils import (removeCollider, removeRigidBody,
                                      setRigidBody)
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F
from tqdm import tqdm, trange

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils_sim import init_capture, write_rgb_data, get_default_import_config
from utils.utils_3d import find_rigid_alignment, quat_diff, quat_diff_batch
from utils.sim_consts import *

from omni.physx import acquire_physx_interface


sns.set()

# Hyperparams
delta_0 = 0.0004
alpha = 0.6

CONTACT_AREAS = {
    "panda": {
        "L": [[-0.008787, 0.000071, 0.036023], [0.008787, 0.000071, 0.036023], [0.008787, 0.000071, 0.053879],
              [-0.008787, 0.000071, 0.053879]],
        "R": [[-0.008787, 0.000071, 0.036023], [0.008787, 0.000071, 0.036023], [0.008787, 0.000071, 0.053879],
              [-0.008787, 0.000071, 0.053879]]
    }
}

sns.set()

PWD = os.path.dirname(os.path.abspath(__file__))

all_grasps = json.load(
    open(f"{PWD}/data/gapartnet/grasps.json", "r"))
all_configs = json.load(
    open(f"{PWD}/data/gapartnet/selected.json", 'r'))
all_grasps = {k: v for k, v in all_grasps.items() if not v == {}}

# TASK_IDS = [ list(all_grasps.keys())[args.idx] ]
TASK_IDS = [str(args.idx)]

quat_diff = lambda q1, q2: 2 * np.arccos(np.abs(np.dot(q1, q2)))
quat_diff_batch = lambda q1, q2: 2 * torch.acos(torch.abs(torch.bmm(q1.unsqueeze(1), q2.unsqueeze(-1)).squeeze(-1)))

if args.capture:
    init_capture()


class Manipulation(BaseTask):
    def __init__(self, i_env, obj_id, friction_mat, report_dir, offset=None, hand_name="panda"):
        super().__init__(name=f"{i_env}_{obj_id}", offset=offset)

        self.object_dir = f"{PWD}/data/gapartnet/{obj_id}"
        self.obj_config = all_configs[obj_id]
        self.time_stamp = str(int(datetime.now().timestamp()))
        self.capture_dir = os.path.join(self.object_dir, f"result-{self.time_stamp}")
        self.report_json = os.path.join(self.object_dir, f"result-{self.time_stamp}.json")
        self.report_pt = os.path.join(self.object_dir, f"result-{self.time_stamp}.pt")

        self.i_env = i_env
        self.hand_name = "panda"
        self.obj_id = str(obj_id)

        self.friction_mat = friction_mat

        self.scene_prim = f"/World/Env_{self.i_env}"
        self.object_prim_path = f"{self.scene_prim}/Obj_{self.obj_id}"
        self.franka_prim_path = f"{self.scene_prim}/Manipulator"

        self.locked_marker_idx = None

        # Data dump
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        self.data = []
        self.attempt_counter = 0

        # Dummy
        self.q = None
        self.hit_wall = False
        self.to_hi, self.to_lo = [], []

        self.x_offset = all_grasps[self.obj_id].get("x_offset", 0.0)
        self.y_offset = all_grasps[self.obj_id].get("y_offset", 0.0)
        self.z_offset = all_grasps[self.obj_id].get("z_offset", 0.0)

        print(f"Manipulating {obj_id}")

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        # Load robot and its IK solver
        self._franka = Franka(prim_path=self.franka_prim_path, name=f"manipulator_{self.i_env}")
        scene.add(self._franka)

        # Load object
        self.target_link = "handle"

        self.base_link = self.obj_config['base_link']
        self.handle_base_link = self.obj_config['target_part']
        self.target_link = self.obj_config['grasp_link']
        self.all_links = self.obj_config['all_links']

        import_config = get_default_import_config()
        urdf_path = os.path.join(self.object_dir, "mobility_relabel_gapartnet.urdf")

        result, prim_path = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf_path,
                                                      import_config=import_config)
        omni.kit.commands.execute("MovePrim", path_from=prim_path, path_to=self.object_prim_path)

        self.target_joint = self.obj_config['target_joint']
        self.target_joint_path = f"{self.object_prim_path}/{self.target_joint}"
        self.target_joint_name = self.target_joint_path.split("/")[-1]

        # Load point clouds for computation
        self.handle_mesh = tm.load(os.path.join(PWD, self.obj_config['grasp_part_mesh']), force='mesh')
        self.handle_pt = torch.tensor(self.handle_mesh.sample(4096), dtype=torch.float32, device='cuda')

        self.l_finger_kpt = torch.tensor(CONTACT_AREAS[self.hand_name]["L"], dtype=torch.float32, device='cuda')
        self.r_finger_kpt = torch.tensor(CONTACT_AREAS[self.hand_name]["R"], dtype=torch.float32, device='cuda')
        self.finger_xx, self.finger_yy = torch.linspace(0, 1, 10, device='cuda'), torch.linspace(0, 1, 10,
                                                                                                 device='cuda')
        self.l_finger_grid, self.r_finger_grid = torch.stack(torch.meshgrid([self.finger_xx, self.finger_yy]),
                                                             dim=-1).reshape(-1, 2).clone(), torch.stack(
            torch.meshgrid([self.finger_xx, self.finger_yy]), dim=-1).reshape(-1, 2).clone()
        self.l_finger_pt = self.l_finger_kpt[0].unsqueeze(0) + self.l_finger_grid[:, 0].unsqueeze(-1) * (
                    self.l_finger_kpt[1] - self.l_finger_kpt[0]).unsqueeze(0) + self.l_finger_grid[:, 1].unsqueeze(
            -1) * (self.l_finger_kpt[3] - self.l_finger_kpt[0]).unsqueeze(0)
        self.r_finger_pt = self.r_finger_kpt[0].unsqueeze(0) + self.r_finger_grid[:, 0].unsqueeze(-1) * (
                    self.r_finger_kpt[1] - self.r_finger_kpt[0]).unsqueeze(0) + self.r_finger_grid[:, 1].unsqueeze(
            -1) * (self.r_finger_kpt[3] - self.r_finger_kpt[0]).unsqueeze(0)

        stage = omni.usd.get_context().get_stage()

        self.hand_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand")
        self.object_prim = stage.GetPrimAtPath(self.object_prim_path)
        self.r_finger_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_rightfinger")
        self.l_finger_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_leftfinger")
        self.finger_joint_prim_1 = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand/panda_finger_joint1")
        self.finger_joint_prim_2 = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand/panda_finger_joint2")
        self.target_joint_prim = stage.GetPrimAtPath(self.target_joint_path)
        self.handle_prim = stage.GetPrimAtPath(f"{self.object_prim_path}/{self.target_link}")
        self.base_link_prim = stage.GetPrimAtPath(f"{self.object_prim_path}/base_link")
        self.target_joint_type = self.obj_config['joint_type']

        self.object_prim.GetAttribute("xformOp:translate").Set(tuple(
            Gf.Vec3f(args.x_offset + self.x_offset, args.y_offset + self.y_offset,
                     self.obj_config['height'] * args.obj_scale + self.z_offset)))
        self.object_prim.GetAttribute("xformOp:scale").Set(
            tuple(Gf.Vec3f(args.obj_scale, args.obj_scale, args.obj_scale)))

        ## Set object position
        self._task_objects["Manipulator"] = self._franka
        self._task_objects["Object"] = XFormPrim(prim_path=self.object_prim_path, name=f"object-{self.i_env}")
        self._move_task_objects_to_their_frame()

        # Goal config
        self.succ_dof_pos = 0.25 if self.target_joint_type == "slider" else np.pi / 3

        finger_drive_1 = UsdPhysics.DriveAPI.Get(self.finger_joint_prim_1, "linear")
        finger_drive_2 = UsdPhysics.DriveAPI.Get(self.finger_joint_prim_2, "linear")
        finger_drive_1.GetMaxForceAttr().Set(1e4)
        finger_drive_2.GetMaxForceAttr().Set(1e4)

        for link in self.all_links:
            if link == self.target_link or link == self.base_link or link == self.handle_base_link:
                UsdPhysics.MassAPI.Apply(get_prim_at_path(f"{self.object_prim_path}/{link}")).CreateMassAttr().Set(0.25)
                continue
            print(f"Removing collider and rigid body for {link}")
            removeCollider(get_prim_at_path(f"{self.object_prim_path}/{link}"))
            removeRigidBody(get_prim_at_path(f"{self.object_prim_path}/{link}"))

        if self.obj_id in ["20043", "32932", "34610", "45243", "45677"]:
            setRigidBody(self.handle_prim, "convexHull", False)

        elif self.obj_id in ["46462", "45756"]:
            setRigidBody(self.handle_prim, "convexDecomposition", False)
        else:
            setRigidBody(self.handle_prim, "boundingCube", False)

        print(f"Loaded {obj_id}")

    def get_observations(self):
        env_obs = {}
        self.q = self._franka.get_joint_positions()

        hand_transf = torch.tensor(
            UsdGeom.Xformable(self.hand_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            device='cuda').transpose(-1, -2)
        r_finger_transf = torch.tensor(
            UsdGeom.Xformable(self.r_finger_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            device='cuda').transpose(-1, -2)
        l_finger_transf = torch.tensor(
            UsdGeom.Xformable(self.l_finger_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            device='cuda').transpose(-1, -2)
        handle_transf = torch.tensor(
            UsdGeom.Xformable(self.handle_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
            device='cuda').transpose(-1, -2)
        l_finger_pt, r_finger_pt, handle_pt = F.pad(self.l_finger_pt.clone(), (0, 1), mode='constant', value=1), F.pad(
            self.r_finger_pt.clone(), (0, 1), mode='constant', value=1), F.pad(self.handle_pt.clone(), (0, 1),
                                                                               mode='constant', value=1)
        l_finger_pt, r_finger_pt, handle_pt = torch.matmul(l_finger_transf, l_finger_pt.transpose(-1, -2)).transpose(-1,
                                                                                                                     -2)[
                                              :, :3], torch.matmul(r_finger_transf,
                                                                   r_finger_pt.transpose(-1, -2)).transpose(-1, -2)[:,
                                                      :3], torch.matmul(handle_transf,
                                                                        handle_pt.transpose(-1, -2)).transpose(-1, -2)[
                                                           :, :3]

        self.marker_pts = r_finger_pt  # Right side markers only

        dist = torch.cdist(handle_pt, self.marker_pts)
        env_obs['finger_to_handle_dist'] = dist

        finger_pts_dist = dist.min(dim=-2)[0]
        finger_marker_contact = finger_pts_dist < CONTACT_THRES

        env_obs['hand_transf'] = hand_transf
        env_obs['r_finger_transf'] = r_finger_transf

        marker_relative_transf = hand_transf

        if self.locked_marker_idx is None and finger_marker_contact.float().mean() > 0.1:
            self.locked_marker_idx = torch.where(finger_marker_contact)[0]
            self.unlocked_marker_idx = torch.where(~finger_marker_contact)[0]
            tqdm.write(
                f"Locking {len(self.locked_marker_idx)} / {len(finger_marker_contact)} marker(s) at {self.locked_marker_idx.cpu().numpy().tolist()}")

            locked_marker_pos = F.pad(self.marker_pts[self.locked_marker_idx].clone(), (0, 1), mode='constant', value=1)
            unlocked_marker_pos = F.pad(self.marker_pts[self.unlocked_marker_idx].clone(), (0, 1), mode='constant',
                                        value=1)
            self.handle_marker_bound_pos = torch.matmul(torch.inverse(handle_transf),
                                                        locked_marker_pos.transpose(-1, -2)).transpose(-1, -2)
            self.unlocked_parker_pos = torch.matmul(torch.inverse(marker_relative_transf),
                                                    unlocked_marker_pos.transpose(-1, -2)).transpose(-1, -2).clone()
            self.init_marker_pos = torch.matmul(torch.inverse(marker_relative_transf),
                                                locked_marker_pos.transpose(-1, -2)).transpose(-1, -2).clone()
            self.kabsch_noise = torch.normal(0, 0.00025, size=self.init_marker_pos.shape, device='cuda')

        if self.locked_marker_idx is not None:
            locked_marker_pos = torch.matmul(handle_transf, self.handle_marker_bound_pos.transpose(-1, -2)).transpose(
                -1, -2)
            curr_marker_pos = torch.matmul(torch.inverse(marker_relative_transf),
                                           locked_marker_pos.transpose(-1, -2)).transpose(-1, -2)

            self.init_marker_pos_world, self.curr_marker_pos_world = torch.matmul(marker_relative_transf,
                                                                                  self.init_marker_pos.transpose(-1, -2)).transpose(-1, -2), locked_marker_pos.clone()
            marker_dspl_r, marker_dspl_t = find_rigid_alignment(self.init_marker_pos_world[:, :3],
                                                                self.curr_marker_pos_world[:, :3])

            marker_dspl_transf = torch.eye(4, device='cuda')
            marker_dspl_transf[:3, :3], marker_dspl_transf[:3, 3] = marker_dspl_r, marker_dspl_t




            env_obs['marker_dspl_r'] = marker_dspl_r
            env_obs['marker_dspl_transf'] = marker_dspl_transf
            env_obs['marker_dspl_dist'] = (curr_marker_pos - self.init_marker_pos).norm(dim=-1).mean()

            self.data.append({
                "unlocked_marker_pos": self.unlocked_parker_pos.clone(),
                "init_marker_pos": self.init_marker_pos.clone(),
                "curr_marker_pos": curr_marker_pos.clone(),
                "cur_dof_pos": self.cur_dof_pos
            })

        env_obs['achieved'] = torch.tensor(self._task_achieved, dtype=torch.float32, device='cuda')

        return {f"{self.i_env}_obs": env_obs}

    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.art = self.dc.get_articulation(self.target_joint_path)
        self.dof_ptr = self.dc.find_articulation_dof(self.art, self.target_joint_name)
        self.cur_dof_pos = self.dc.get_dof_position(self.dof_ptr)
        self._task_achieved = self.cur_dof_pos > self.succ_dof_pos

        if self._task_achieved:
            torch.save(self.data, self.report_pt)

            simulation_app.close()
            exit()

    def report(self, succ: bool, hit_wall: bool, simulation_time: float):
        json.dump(
            {
                "succ": succ,
                "q": self.cur_dof_pos,
                "hit_wall_termination": bool(hit_wall),
                "n_it": self.attempt_counter,
                "time": simulation_time
            }, open(self.report_json, 'w')
        )

        tqdm.write(f"Reported result (succ: {succ}) to {self.report_json}.")

    def post_reset(self):
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self.art = self.dc.get_articulation(self.target_joint_path)
        self.dc.wake_up_articulation(self.art)
        self.dof_ptr = self.dc.find_articulation_dof(self.art, self.target_joint_name)

        self._task_achieved = False
        self.attempt_counter = 0

    def _setup_physics_material(self, path, physics_material_path):
        stage = omni.usd.get_context().get_stage()
        collisionAPI = UsdPhysics.CollisionAPI.Get(stage, path)
        prim = stage.GetPrimAtPath(path)
        if not collisionAPI:
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
        # apply material
        physicsUtils.add_physics_material_to_prim(stage, prim, physics_material_path)

    def setup_ik_solver(self, franka):
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kine_solver = LulaKinematicsSolver(**kinematics_config)
        self._art_kine_solver = ArticulationKinematicsSolver(self._franka, self._kine_solver, "right_gripper")

    def set_grasp_pose(self):
        p, r = all_grasps[self.obj_id]['p'], all_grasps[self.obj_id]['R']
        p = np.asarray(p) * args.obj_scale
        p[0] += args.x_offset + self.x_offset
        p[1] += args.y_offset + self.y_offset
        p[0] += 0.046
        p[2] += self.obj_config['height'] * args.obj_scale + self.z_offset
        r = R.from_euler("ZYX", r, False).as_quat()  # [[3, 0, 1, 2]]

        self.grasp_p = p

        print(f"Setting grasp pose to {p}, {r}")

        robot_base_translation, robot_base_orientation = self._franka.get_world_pose()
        self._kine_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        action, ik_success = self._art_kine_solver.compute_inverse_kinematics(p + self._offset, r)

        if ik_success:
            action.joint_positions[-1] = 0.04
            action.joint_positions[-2] = 0.04
            self._franka.set_joint_positions(action.joint_positions)
            self._franka.set_joint_velocities([0.0] * 9)

        else:
            tqdm.write("IK failed")

        self._franka.gripper.apply_action(ArticulationAction([0.00, 0.00]))

        return ik_success

    def do_ik(self, p, r):
        return self._art_kine_solver.compute_inverse_kinematics(p + self._offset, r)


world = World()
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()
world._physics_context.set_gravity(value=0.0)

stage.GetPrimAtPath("/World/defaultGroundPlane/Enviroment").GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, -1.0))
stage.GetPrimAtPath("/World/defaultGroundPlane/GroundPlane").GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, -1.0))

_kinematics_solver = None
_articulation_kinematics_solver = None
n_tasks = len(TASK_IDS)

_tasks = []
_frankas = []
_rmpflows = []
_art_rmpflows = []
_rmpflow_controllers = []
_art_controllers = []
target_joints = []
target_joint_drive = []
taget_links = []
_articulation_kinematics_solvers = []

# 0 - Exploration, 1 - Modification, 2 - Finished.
env_states = torch.zeros([n_tasks], dtype=torch.int32, device='cuda')

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find nucleus server with /Isaac folder")

draw = _debug_draw.acquire_debug_draw_interface()

n_each_row = 3
spacing = 2.0

time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
data_dir = os.path.join(PWD, "data", "manipulation", time_tag)


light_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
light_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0, -3.0, 6.0))
light_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.015, 0.015, 0.015))

_material_static_friction = 1.0
_material_dynamic_friction = 1.0
_material_restitution = 0.0
_physicsMaterialPath = None

if _physicsMaterialPath is None:
    _physicsMaterialPath = stage.GetPrimAtPath("/World").GetPath().AppendChild("physicsMaterial")
    UsdShade.Material.Define(stage, _physicsMaterialPath)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(_physicsMaterialPath))
    material.CreateStaticFrictionAttr().Set(_material_static_friction)
    material.CreateDynamicFrictionAttr().Set(_material_dynamic_friction)
    material.CreateRestitutionAttr().Set(_material_restitution)

for i_task, obj_id in enumerate(TASK_IDS):
    task_data_dir = os.path.join(data_dir, f"{i_task}_{obj_id}")
    stage.DefinePrim(f"/World/Env_{i_task}", "Xform")
    world.add_task(Manipulation(i_task, obj_id, _physicsMaterialPath, task_data_dir,
                                offset=np.array([i_task // n_each_row, i_task % n_each_row, 0.0]) * spacing))

mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

world.reset()
# Initialize an RmpFlow object
for i_task, obj_id in enumerate(TASK_IDS):
    _tasks.append(world.get_task(name=f"{i_task}_{obj_id}"))

    _frankas.append(_tasks[i_task]._franka)
    _art_controllers.append(_frankas[i_task].get_articulation_controller())
    target_joints.append(_tasks[i_task].target_joint_prim)

    _rmpflows.append(RmpFlow(
        robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
        urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
        rmpflow_config_path=rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml",
        end_effector_frame_name="panda_hand",
        maximum_substep_size=0.00334
    ))
    _art_rmpflows.append(ArticulationMotionPolicy(_frankas[i_task], _rmpflows[i_task]))
    _rmpflow_controllers.append(RMPFlowController(name=f"controller_{i_task}", robot_articulation=_frankas[i_task]))

    f = world.scene.get_object(f"manipulator_{i_task}")
    _tasks[i_task].setup_ik_solver(f)
    f.disable_gravity()


target_joint_drive = [UsdPhysics.DriveAPI.Apply(target_joints[j].GetPrim(),
                                                "linear" if _tasks[j].target_joint_type == "slider" else "angular") for
                      j in range(len(target_joints))]



def lock_joint_drive(lock_idx: list):
    for i in lock_idx:
        target_joint_drive[i].GetDampingAttr().Set(1e8)
        target_joint_drive[i].GetMaxForceAttr().Set(1e8)


def release_joint_drive(release_idx: list):
    for i in release_idx:
        target_joint_drive[i].GetDampingAttr().Set(50.0)
        target_joint_drive[i].GetMaxForceAttr().Set(1e-4)


def get_proceeding_dir(hand_transf):
    dir = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda').unsqueeze(0).tile([n_tasks, 1])
    dir = torch.matmul(hand_transf[:, :3, :3], dir.unsqueeze(-1)).squeeze(-1)
    return dir


def fetch_obs():
    world_obs = world.get_observations()
    obs = defaultdict(list)

    for i in range(n_tasks):
        if f"{i}_obs" not in world_obs:
            continue

        for k, v in world_obs[f"{i}_obs"].items():
            obs[k].append(v)

    return {k: torch.stack(v, dim=0) for k, v in obs.items()}


def rmpflow_action(target_pos, target_rot, visualize=False):
    for i in range(len(target_pos)):
        f, c, r = _frankas[i], _art_rmpflows[i], _rmpflows[i]
        r.set_end_effector_target(target_pos[i], target_rot[i, [3, 0, 1, 2]])
        actions = c.get_next_articulation_action(1 / 60)
        f.apply_action(actions)
        if recv_flag[i]:
            f.gripper.apply_action(ArticulationAction(_tasks[i].grasp_q))
        else:
            f.gripper.apply_action(ArticulationAction([0.0, 0.0]))


recv_stuck_count = torch.zeros([n_tasks], dtype=torch.int32, device='cuda')
lock_joint_drive(lock_idx=list(range(n_tasks)))

for i_task, t in enumerate(_tasks):
    _art_controllers[i_task].apply_action(ArticulationAction([0.0] * 9))
    ik_success = t.set_grasp_pose()

    if not ik_success:
        simulation_app.close()
        exit()

step_record_start = 10
record_interval = 1
step_start = 100

curr_proceed_dir, curr_proceed_base_transf = torch.zeros([n_tasks, 3], dtype=torch.float32, device='cuda'), torch.zeros(
    [n_tasks, 4, 4], dtype=torch.float32, device='cuda')
next_proc = torch.tensor([], dtype=torch.long, device='cuda')
render_path = f"{_tasks[0].capture_dir}/capture"



for step in trange(MAX_STEPS + 1):
    world.step(render=args.capture or (not args.headless))

    if args.capture and step == step_record_start:
        cam = rep.create.camera(position=(-1.0, 1.0, 1.25), look_at=tuple(_tasks[0].grasp_p))
        rp = rep.create.render_product(cam, (3840, 2160))

        print(f"Outputting data to {render_path}.")
        os.makedirs(render_path, exist_ok=True)

        writer = rep.WriterRegistry.get("BasicWriter")

        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach(rp)

    if args.capture and step > step_record_start and step % record_interval == 0:
        rep.orchestrator.step(rt_subframes=4, pause_timeline=False)
        write_rgb_data(rgb_annot.get_data(), f"{render_path}/Step-{step}")

    if step == step_start:
        release_joint_drive(release_idx=list(range(n_tasks)))
        obs = fetch_obs()
        proc_idx = torch.where(env_states == STATE_PROC)[0]

        curr_proceed_base_transf = obs['hand_transf'].clone()
        curr_proceed_dir = get_proceeding_dir(curr_proceed_base_transf)

        for t in _tasks:
            t.grasp_q = t._franka.get_joint_positions()[-2:].tolist()
        continue

    if step < step_start:
        for t in _tasks:
            t._franka.gripper.apply_action(ArticulationAction([0.00, 0.00]))
        continue

    if step == MAX_STEPS:
        for t in _tasks:
            if t._task_achieved:
                continue
        break

    proc_flag = env_states == STATE_PROC
    proc_idx = torch.where(proc_flag)[0]
    succ_idx = torch.where(env_states == STATE_SUCC)[0]
    recv_flag = env_states == STATE_RECV

    obs = fetch_obs()
    hand_transf = obs['hand_transf']

    if not "marker_dspl_r" in obs:
        continue

    _tasks[0].data[-1]['step'] = step
    _tasks[0].data[-1]['state'] = "RECV" if env_states[0] == STATE_RECV else "PROC"

    # Compute target transformation for RMPFlow Control
    target_transf = hand_transf.clone()
    marker_dspl_r, marker_dspl_transf, marker_dspl_dist = obs['marker_dspl_r'], obs['marker_dspl_transf'], obs[
        'marker_dspl_dist']

    ## Proceed targets
    curr_proceed_base_transf[next_proc] = hand_transf[next_proc].clone()
    curr_proceed_dir[next_proc] = get_proceeding_dir(curr_proceed_base_transf[next_proc])

    if len(proc_idx) > 0:
        curr_proceed_base_transf[proc_idx, :3, 3] = curr_proceed_base_transf[proc_idx, :3, 3] + curr_proceed_dir[
            proc_idx] * 0.0001
        target_transf[proc_idx] = curr_proceed_base_transf[proc_idx].clone()

    ## Recovery targets
    recv_angle = np.abs(R.from_matrix(marker_dspl_r.cpu().numpy()).as_euler("XYZ", True)).sum(axis=-1)
    recv_accept = (torch.det(marker_dspl_r) > 0.9999)
    recv_idx = torch.where(recv_flag * recv_accept)[0]

    recv_stuck_count = recv_stuck_count + 1
    recv_stuck_count[succ_idx] = 0
    recv_stuck_count[proc_idx] = 0
    if len(recv_idx) > 0:
        target_transf[recv_idx] = torch.matmul(marker_dspl_transf[recv_idx], hand_transf[recv_idx]).float()

    recv_stuck_idx = torch.where(recv_stuck_count % 250 == 249)[0]
    if len(recv_stuck_idx) > 0:
        target_transf[recv_stuck_idx] += torch.normal(0, 0.01, size=target_transf[recv_stuck_idx].shape, device='cuda')
        tqdm.write(f"Stuck at simulation step {step}, applying noise.")

    really_stuck_idx = torch.where(recv_stuck_count % 2500 == 2499)[0]
    if len(really_stuck_idx) > 0:
        for i in really_stuck_idx:
            ik_transf = hand_transf[i].clone()
            ik_transf[:3, 3] += curr_proceed_dir[i] * 0.1

            ik_p, ik_r = ik_transf[:3, 3].cpu().numpy(), ik_transf[:3, :3].cpu().numpy()
            ik_r = R.from_matrix(ik_r).as_quat()  # [[3, 0, 1, 2]]
            a, s = _tasks[i].do_ik(ik_p, ik_r)

            if s:
                print(f"Really stuck at simulation step {step}, IK validation failed.")
                # _tasks[i].report(False, False, 1e6)
                simulation_app.close()
                exit()
            else:
                print(f"Really stuck at simulation step {step}, IK validation passed. Counted as successfult.")
                _tasks[i].hit_wall = True

    # Update env states
    next_recv = torch.where(proc_flag * (marker_dspl_dist > delta_0))[0] 
    next_proc = torch.where(recv_flag * (marker_dspl_dist < delta_0 * alpha))[0]  
    next_succ_idx = torch.where(obs['achieved'] == 1.0)[0]

    env_states[next_proc] = STATE_PROC
    env_states[next_recv] = STATE_RECV
    env_states[next_succ_idx] = STATE_SUCC  # Must come last to override.

    for i_proc in next_proc:
        _tasks[i_proc].attempt_counter += 1

    lock_joint_drive(lock_idx=next_recv.cpu().numpy().tolist())
    release_joint_drive(release_idx=next_proc.cpu().numpy().tolist())

    # Apply RMPFlow control
    target_pos, target_rot = target_transf[:, :3, 3].cpu().numpy(), target_transf[:, :3, :3].cpu().numpy()
    target_rot = R.from_matrix(target_rot).as_quat()
    rmpflow_action(target_pos, target_rot)

    if step % 100 == 99:
        tqdm.write(f"Env 0: Handle-to-target distance: {_tasks[0].cur_dof_pos}")
        
simulation_app.close()
