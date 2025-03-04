import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--n_ctrl", default="5", type=int)
parser.add_argument("--id", default="15", type=int)
parser.add_argument("--inverse", action='store_true')
parser.add_argument("--headless", action='store_true')
parser.add_argument("--capture", action="store_true")
parser.add_argument("--report", action="store_true")
parser.add_argument("--x_offset", default="0.5", type=float)
parser.add_argument("--y_offset", default="0.0", type=float)
parser.add_argument("--z_offset", default="0.0", type=float)

args = parser.parse_args()

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": args.headless}) # we can also run as headless.

import csv
import json
import os
from collections import defaultdict
from datetime import datetime
import carb
import numpy as np
import seaborn as sns
import torch
import trimesh as tm
from tqdm import tqdm, trange
import seaborn as sns
import omni.replicator.core as rep
import bezier
import omni.kit
import omni.graph.action
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.motion_generation import (ArticulationKinematicsSolver,
                                          ArticulationMotionPolicy,
                                          LulaKinematicsSolver, RmpFlow,
                                          interface_config_loader)
from omni.physx.scripts import physicsUtils
from omni.physx.scripts.utils import removeCollider, removeRigidBody, setCollider
from pxr import Gf, Usd, UsdGeom, UsdShade, UsdPhysics
from scipy.spatial.transform import Rotation as R
from torch.nn import functional as F

from omni.physx import acquire_physx_interface
# physx = acquire_physx_interface()
# physx.overwrite_gpu_setting(1)

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils_sim import init_capture, write_rgb_data, get_default_import_config
from utils.utils_3d import find_rigid_alignment, quat_diff, quat_diff_batch
from utils.sim_consts import *


sns.set()

# Hyperparams
W, H = 0.68, 0.48

PWD = os.path.dirname(os.path.abspath(__file__))
TASK_IDS = [ (args.n_ctrl, args.id, args.inverse) ]

if args.n_ctrl == 5 and args.id == 15:
    args.x_offset = 0.6
if args.n_ctrl == 3 and args.id == 15:
    args.x_offset = 0.4
OBJ_X_OFST = args.x_offset
OBJ_Y_OFST = args.y_offset
OBJ_Z_OFST = args.z_offset
SUCC_HANDLE_RANGE = 0.01

class Manipulation(BaseTask):
    def __init__(self, i_env, n_pts, curve_id, inverse, _physicsMaterialPath, offset=None, hand_name="panda"):
        super().__init__(name=f"{i_env}_{n_pts}_{curve_id}", offset=offset)
        
        self.n_pts = n_pts
        self.curve_id = curve_id
        self.object_dir = f"{PWD}/data/playboard/{n_pts}/{curve_id}"
        self.curve_config = json.load(open(os.path.join(self.object_dir, "control_points.json"), 'r'))
        
        self.time_stamp = str(int(datetime.now().timestamp()))
        if args.inverse:
            self.time_stamp += "-inverse"
        self.capture_dir = os.path.join(self.object_dir, f"result-{self.time_stamp}")
        self.report_json = os.path.join(self.object_dir, f"result-{self.time_stamp}.json")
        self.report_pt = os.path.join(self.object_dir, f"result-{self.time_stamp}.pt")
        
        self.i_env = i_env
        self.hand_name = "panda"
        self.obj_id = f"{n_pts}_{curve_id}"
        self.inverse = inverse
        
        self._physicsMaterialPath = _physicsMaterialPath
        
        self.scene_prim = f"/World/Env_{self.i_env}"
        self.object_prim_path = f"{self.scene_prim}/Object_{self.obj_id}"
        self.franka_prim_path = f"{self.scene_prim}/Manipulator"
        
        self.locked_marker_idx = None
        
        # Data dump
        self.data = []
        self.attempt_counter = 0
        
        
        # Dummy
        self.q = None
        self.hit_wall = False
        self.to_hi, self.to_lo = [], []
        
        print(f"Manipulating {obj_id}")
        
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        
        # Load robot and its IK solver
        self._franka = Franka(prim_path=self.franka_prim_path, name=f"manipulator_{self.i_env}")
        scene.add(self._franka)
        
        wrist_joint = UsdPhysics.PrismaticJoint(get_prim_at_path(f"{self.franka_prim_path}/panda_link6/panda_joint7"))
        wrist_joint.GetUpperLimitAttr().Set(720.0)
        wrist_joint.GetLowerLimitAttr().Set(-720.0)

        # Load object
        self.target_link = "handle"

        self.all_links = [ "base_link", "handle" ]
        
        import_config = get_default_import_config()
        urdf_path = os.path.join(self.object_dir, "playboard_inv.urdf" if self.inverse else "playboard.urdf")
        self.control_points = np.asarray(json.load(open(os.path.join(self.object_dir, "control_points.json"), 'r'))['control_points']) / 100
        self.control_points[:, 0] -= W/2
        self.control_points[:, 1] -= H/2
        self.control_points = np.matmul(R.from_euler("XYZ", [0.0, 0.0, 90.017], degrees=True).as_matrix()[:2, :2], self.control_points.T).T.squeeze()
        self.control_points[:, 0] += OBJ_X_OFST
        self.control_points[:, 1] += OBJ_Y_OFST
        self.curve = bezier.Curve(
            nodes=self.control_points.T,
            degree=len(self.control_points) - 1,
        )
        
        result, prim_path = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config)
        omni.kit.commands.execute("MovePrim", path_from=prim_path, path_to=self.object_prim_path)
        
        self.target_joints = [ "virtual/pri_x", "virtual_x/pri_y", "virtual_xy/rev_z" ]
        self.target_joint_paths = [ f"{self.object_prim_path}/{j}" for j in self.target_joints ]
        self.target_joint_names = [ p.split("/")[-1] for p in self.target_joint_paths ]
                
        # Load point clouds for computation
        self.handle_mesh = tm.load(os.path.join(PWD, "data/playboard/handle/train_elevated.stl"), force='mesh')
        self.handle_pt = torch.tensor(self.handle_mesh.sample(4096), dtype=torch.float32, device='cuda')
        
        self.l_finger_kpt = torch.tensor(CONTACT_AREAS[self.hand_name]["L"], dtype=torch.float32, device='cuda')
        self.r_finger_kpt = torch.tensor(CONTACT_AREAS[self.hand_name]["R"], dtype=torch.float32, device='cuda')
        self.finger_xx, self.finger_yy = torch.linspace(0, 1, 10, device='cuda'), torch.linspace(0, 1, 10, device='cuda')
        self.l_finger_grid, self.r_finger_grid = torch.stack(torch.meshgrid([self.finger_xx, self.finger_yy]), dim=-1).reshape(-1, 2).clone(), torch.stack(torch.meshgrid([self.finger_xx, self.finger_yy]), dim=-1).reshape(-1, 2).clone()
        self.l_finger_pt = self.l_finger_kpt[0].unsqueeze(0) + self.l_finger_grid[:, 0].unsqueeze(-1) * (self.l_finger_kpt[1] - self.l_finger_kpt[0]).unsqueeze(0) + self.l_finger_grid[:, 1].unsqueeze(-1) * (self.l_finger_kpt[3] - self.l_finger_kpt[0]).unsqueeze(0)
        self.r_finger_pt = self.r_finger_kpt[0].unsqueeze(0) + self.r_finger_grid[:, 0].unsqueeze(-1) * (self.r_finger_kpt[1] - self.r_finger_kpt[0]).unsqueeze(0) + self.r_finger_grid[:, 1].unsqueeze(-1) * (self.r_finger_kpt[3] - self.r_finger_kpt[0]).unsqueeze(0)
        
        stage = omni.usd.get_context().get_stage()
        
        self.hand_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand")
        self.object_prim = stage.GetPrimAtPath(self.object_prim_path)
        self.r_finger_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_rightfinger")
        self.l_finger_prim = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_leftfinger")
        self.finger_joint_prim_1 = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand/panda_finger_joint1")
        self.finger_joint_prim_2 = stage.GetPrimAtPath(f"{self.franka_prim_path}/panda_hand/panda_finger_joint2")
        self.target_joint_prims = [ stage.GetPrimAtPath(j) for j in self.target_joint_paths ]
        self.handle_prim = stage.GetPrimAtPath(f"{self.object_prim_path}/{self.target_link}") 
        self.base_link_prim = stage.GetPrimAtPath(f"{self.object_prim_path}/base_link") 
        
        self.object_prim.GetAttribute("xformOp:translate").Set(tuple(Gf.Vec3f(OBJ_X_OFST, OBJ_Y_OFST, OBJ_Z_OFST)))
        self.object_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0.707, Gf.Vec3d(0.0, 0.0, 0.707)))
        
        ## Set object position
        self._task_objects["Manipulator"] = self._franka
        self._task_objects["Object"] = XFormPrim(prim_path=self.object_prim_path, name=f"object-{self.i_env}")
        self._move_task_objects_to_their_frame()
        
        # Goal config
        self.succ_handle_pos = np.asarray(self.curve_config["terminal_inner"][0]) if self.inverse else np.asarray(self.curve_config["terminal_inner"][1])
        self.succ_handle_pos = self.succ_handle_pos / 100
        self.succ_handle_pos[0] -= (W / 2)
        self.succ_handle_pos[1] -= (H / 2)
        self.succ_handle_pos = np.matmul(R.from_euler("XYZ", [0.0, 0.0, 90.017], degrees=True).as_matrix()[:2, :2], self.succ_handle_pos[:, None]).squeeze()
        self.succ_handle_pos[0] += OBJ_X_OFST
        self.succ_handle_pos[1] += OBJ_Y_OFST
        
        finger_drive_1 = UsdPhysics.DriveAPI.Get(self.finger_joint_prim_1, "linear")
        finger_drive_2 = UsdPhysics.DriveAPI.Get(self.finger_joint_prim_2, "linear")
        finger_drive_1.GetMaxForceAttr().Set(1e4)
        finger_drive_2.GetMaxForceAttr().Set(1e4)
        
        removeCollider(get_prim_at_path(f"{self.object_prim_path}/base_link"))
        removeRigidBody(get_prim_at_path(f"{self.object_prim_path}/base_link"))
        setCollider(get_prim_at_path(f"{self.object_prim_path}/base_link"), "sdf")
        
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(f"{self.object_prim_path}/handle/collisions/mesh_0"), self._physicsMaterialPath)
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(f"{self.object_prim_path}/handle/collisions/mesh_1"), self._physicsMaterialPath)
        physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(f"{self.object_prim_path}/base_link"), self._physicsMaterialPath)
        
        
    def get_observations(self):
        env_obs = {}
        self.q = self._franka.get_joint_positions()

        hand_transf = torch.tensor(UsdGeom.Xformable(self.hand_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), device='cuda').transpose(-1, -2)
        r_finger_transf = torch.tensor(UsdGeom.Xformable(self.r_finger_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), device='cuda').transpose(-1, -2)
        l_finger_transf = torch.tensor(UsdGeom.Xformable(self.l_finger_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), device='cuda').transpose(-1, -2)
        handle_transf = torch.tensor(UsdGeom.Xformable(self.handle_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), device='cuda').transpose(-1, -2)
        l_finger_pt, r_finger_pt, handle_pt = F.pad(self.l_finger_pt.clone(), (0, 1), mode='constant', value=1), F.pad(self.r_finger_pt.clone(), (0, 1), mode='constant', value=1), F.pad(self.handle_pt.clone(), (0, 1), mode='constant', value=1)
        l_finger_pt, r_finger_pt, handle_pt = torch.matmul(l_finger_transf, l_finger_pt.transpose(-1, -2)).transpose(-1, -2)[:, :3], torch.matmul(r_finger_transf, r_finger_pt.transpose(-1, -2)).transpose(-1, -2)[:, :3], torch.matmul(handle_transf, handle_pt.transpose(-1, -2)).transpose(-1, -2)[:, :3]
        
        self.marker_pts = r_finger_pt # Right side markers only
        
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
            locked_marker_pos = F.pad(self.marker_pts[self.locked_marker_idx].clone(), (0, 1), mode='constant', value=1)
            unlocked_marker_pos = F.pad(self.marker_pts[self.unlocked_marker_idx].clone(), (0, 1), mode='constant', value=1)
            self.handle_marker_bound_pos = torch.matmul(torch.inverse(handle_transf), locked_marker_pos.transpose(-1, -2)).transpose(-1, -2)
            self.unlocked_parker_pos = torch.matmul(torch.inverse(marker_relative_transf), unlocked_marker_pos.transpose(-1, -2)).transpose(-1, -2).clone()
            self.init_marker_pos = torch.matmul(torch.inverse(marker_relative_transf), locked_marker_pos.transpose(-1, -2)).transpose(-1, -2).clone()
            self.kabsch_noise = torch.normal(0, 0.001, size=self.init_marker_pos.shape, device='cuda')
            
        if self.locked_marker_idx is not None:
            locked_marker_pos = torch.matmul(handle_transf, self.handle_marker_bound_pos.transpose(-1, -2)).transpose(-1, -2)
            curr_marker_pos = torch.matmul(torch.inverse(marker_relative_transf), locked_marker_pos.transpose(-1, -2)).transpose(-1, -2)
            
            self.init_marker_pos_world, self.curr_marker_pos_world = torch.matmul(marker_relative_transf, self.init_marker_pos.transpose(-1, -2)).transpose(-1, -2), locked_marker_pos.clone()
            marker_dspl_r, marker_dspl_t = find_rigid_alignment(self.init_marker_pos_world[:, :3], self.curr_marker_pos_world[:, :3])
           
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
        
        return { f"{self.i_env}_obs": env_obs }
    
    # Called before each physics step,
    # for instance we can check here if the task was accomplished by
    # changing the color of the cube once its accomplished
    def pre_step(self, control_index, simulation_time):
        self.cur_dof_pos = np.asarray(UsdGeom.Xformable(self.handle_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T[:2, 3]
        self.to_target_dist = np.linalg.norm(self.cur_dof_pos - self.succ_handle_pos)
        self._task_achieved = self.to_target_dist < SUCC_HANDLE_RANGE
        
        if self.q is not None and simulation_time > 20.0:
            self.to_hi, self.to_lo = ARM_DOF_HI - self.q[:-2], self.q[:-2] - ARM_DOF_LO
            hit_hi_wall = ((self.to_hi / ARM_DOF_RG) < 0.01)
            hit_lo_wall = ((self.to_lo / ARM_DOF_RG) < 0.01)
            self.hit_wall = (hit_lo_wall.astype(float).sum() + hit_hi_wall.astype(float).sum()) > 0 or self.hit_wall
            self._task_achieved = self._task_achieved #or self.hit_wall
            
        if self._task_achieved:
            if args.report:
                self.report(True, self.hit_wall, simulation_time)
            torch.save(self.data, self.report_pt)


            simulation_app.close()
            exit()
            
        self._task_achieved = self._task_achieved
        
    def report(self, succ: bool, hit_wall:bool, simulation_time: float):
        json.dump(
            {
                "succ": succ,
                "q": list(self.cur_dof_pos),
                "to_target_dist": self.to_target_dist,
                "hit_wall_termination": bool(hit_wall),
                "n_it": self.attempt_counter,
                "time": simulation_time
            }, open(self.report_json, 'w')
        )
        tqdm.write(f"Reported result (succ: {succ}) to {self.report_json}.")
        
        
    def post_reset(self):
        self._task_achieved = False
        self.data = []
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
        p, r = np.asarray([0.004304, 0.0, 0.163565]), np.asarray([180, 0, 0])
        grasp_transf, grasp_transf = np.eye(4), np.eye(4)
        grasp_transf[:3, :3] = R.from_euler("XYZ", r, True).as_matrix()
        grasp_transf[:3, 3] = p
        handle_transf = np.asarray(UsdGeom.Xformable(self.handle_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        grasp_transf = np.matmul(handle_transf, grasp_transf)
        p = grasp_transf[:3, 3]
        r = R.from_matrix(grasp_transf[:3, :3]).as_quat()[[3, 0, 1, 2]]
        
        robot_base_translation, robot_base_orientation = self._franka.get_world_pose()
        self._kine_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        action, ik_success = self._art_kine_solver.compute_inverse_kinematics(p + self._offset, r)

        if ik_success:
            action.joint_positions[-1] = 0.04
            action.joint_positions[-2] = 0.04
            self._franka.set_joint_positions(action.joint_positions)
            self._franka.set_joint_velocities([0.0] * 9)
            
        else:
            tqdm.write(f"IK failed for object {self.obj_id}")
                
        self._franka.gripper.apply_action(ArticulationAction([0.0, 0.0]))
        
        return ik_success

world = World()
world.scene.add_default_ground_plane()

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
target_joint_drives = []
taget_links = []
_articulation_kinematics_solvers = []

# 0 - Exploration, 1 - Modification, 2 - Finished.
env_states = torch.zeros([n_tasks], dtype=torch.int32, device='cuda')
    
draw = _debug_draw.acquire_debug_draw_interface()

stage = omni.usd.get_context().get_stage()
n_each_row = 3
spacing = 2.0

time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")

stage = omni.usd.get_context().get_stage()

light_prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
light_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0, -3.0, 5.0))
light_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.015, 0.015, 0.015))

# Physics
_material_static_friction = 0.0001
_material_dynamic_friction = 0.0001
_material_restitution = 0.0
_physicsMaterialPath = None

if _physicsMaterialPath is None:
    _physicsMaterialPath = stage.GetPrimAtPath("/World").GetPath().AppendChild("physicsMaterial")
    UsdShade.Material.Define(stage, _physicsMaterialPath)
    material = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(_physicsMaterialPath))
    material.CreateStaticFrictionAttr().Set(_material_static_friction)
    material.CreateDynamicFrictionAttr().Set(_material_dynamic_friction)
    material.CreateRestitutionAttr().Set(_material_restitution)

for i_task, (n_pts, curve_id, inverse) in enumerate(TASK_IDS):
    obj_id = f"{n_pts}_{curve_id}"
    stage.DefinePrim(f"/World/Env_{i_task}", "Xform")
    world.add_task(Manipulation(i_task, n_pts, curve_id, inverse, _physicsMaterialPath, offset=np.array([i_task // n_each_row, i_task % n_each_row, 0.0]) * spacing))

rmp_config_dir = os.path.join(PWD, "data", "franka_omniwrist_lula")

world.reset()
#Initialize an RmpFlow object        
for i_task, (n_pts, curve_id, inverse) in enumerate(TASK_IDS):
    obj_id = f"{n_pts}_{curve_id}"
    _tasks.append(world.get_task(name=f"{i_task}_{obj_id}"))
    
    _frankas.append(_tasks[i_task]._franka)
    _art_controllers.append(_frankas[i_task].get_articulation_controller())
    target_joints += _tasks[i_task].target_joint_prims
    
    _rmpflows.append(RmpFlow(
        robot_description_path = rmp_config_dir + "/rmpflow/robot_descriptor.yaml",
        urdf_path = rmp_config_dir + "/lula_franka_gen.urdf",
        rmpflow_config_path = rmp_config_dir + "/rmpflow/franka_rmpflow_common.yaml",
        end_effector_frame_name = "panda_hand",
        maximum_substep_size = 0.00334
    ))
    
    _art_rmpflows.append(ArticulationMotionPolicy(_frankas[i_task], _rmpflows[i_task]))
    _rmpflow_controllers.append(RMPFlowController(name=f"controller_{i_task}", robot_articulation=_frankas[i_task]))
    
    _rmpflows[-1].reset()
    f = world.scene.get_object(f"manipulator_{i_task}")
    _tasks[i_task].setup_ik_solver(f)
    f.disable_gravity()
    
target_joint_drives = [ UsdPhysics.DriveAPI.Apply(target_joints[j].GetPrim(), "angular" if j % 3 == 2 else "linear") for j in range(len(target_joints)) ]
    
def lock_joint_drive(lock_idx: list):
    for i in lock_idx:
        for j in range(3):
            target_joint_drives[i*3+j].GetDampingAttr().Set(1e8)
            target_joint_drives[i*3+j].GetMaxForceAttr().Set(1e8)

def release_joint_drive(release_idx: list):
    for i in release_idx:
        for j in range(3):
            target_joint_drives[i*3+j].GetDampingAttr().Set(50.0)
            target_joint_drives[i*3+j].GetMaxForceAttr().Set(1e-4)
    
def get_proceeding_dir(hand_transf):
    dir = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32, device='cuda').unsqueeze(0).tile([n_tasks, 1])
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
                
    obs = { k: torch.stack(v, dim=0) for k, v in obs.items() }
    return obs

def rmpflow_action(target_pos, target_rot, visualize=False):
    for i in range(len(target_pos)):
        f, c, r = _frankas[i], _art_rmpflows[i], _rmpflows[i]
        r.set_end_effector_target(target_pos[i], target_rot[i, [3, 0, 1, 2]])
        actions = c.get_next_articulation_action(1 / 30)
        f.apply_action(actions)
        if recv_flag[i]:
            f.gripper.apply_action(ArticulationAction(_tasks[i].grasp_q))
        else:
            f.gripper.apply_action(ArticulationAction([0.0, 0.0]))

recv_stuck_count = torch.zeros([n_tasks], dtype=torch.int32, device='cuda')
step_start = 100

for i_task, t in enumerate(_tasks):
    _art_controllers[i_task].apply_action(ArticulationAction([0.0] * 9))
    ik_success = t.set_grasp_pose()
    
    robot_base_translation, robot_base_orientation = t._franka.get_world_pose()
    _rmpflows[i_task].set_robot_base_pose(robot_base_translation, robot_base_orientation)
        
curr_proceed_dir, curr_proceed_base_transf = torch.zeros([n_tasks, 3], dtype=torch.float32, device='cuda'), torch.zeros([n_tasks, 4, 4], dtype=torch.float32, device='cuda')
next_proc = torch.tensor([], dtype=torch.long, device='cuda')
render_path = f"{_tasks[0].capture_dir}/capture"

step_record_start = 10
record_interval = 5
step_start = 100

if args.capture:
    init_capture()

for step in trange(MAX_STEPS+1):
    if args.capture and step == step_record_start:
        cam = rep.create.camera(position=(1.6, 0.0, 1.0), look_at=tuple([OBJ_X_OFST / 2, OBJ_Y_OFST, OBJ_Z_OFST * 2]))
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
        for t in _tasks:
            t.grasp_q = t._franka.get_joint_positions()[-2:].tolist()
            
        world.step(render=args.capture or (not args.headless))
        continue
                                
    if step == step_start:
        obs = fetch_obs()
        proc_idx = torch.where(env_states == STATE_PROC)[0]
        curr_proceed_base_transf = obs['hand_transf'].clone()
        curr_proceed_dir = get_proceeding_dir(curr_proceed_base_transf)
        continue
    
    if step < step_start:
        world.step(render=args.capture or (not args.headless))
        continue
    
    if step == MAX_STEPS:
        for t in _tasks:
            if t._task_achieved:
                continue
            if args.report:
                t.report(False, False, 1e9)
        break
    
    proc_flag = env_states == STATE_PROC
    proc_idx = torch.where(proc_flag)[0]
    succ_idx = torch.where(env_states == STATE_SUCC)[0]
    recv_flag = env_states == STATE_RECV
    
    obs = fetch_obs()
    hand_transf = obs['hand_transf']
    
    curr_proceed_base_transf[next_proc] = hand_transf[next_proc].clone()
    curr_proceed_dir[next_proc] = get_proceeding_dir(curr_proceed_base_transf[next_proc])

    if "marker_dspl_r" not in obs:
        world.step(render=args.capture or (not args.headless))
        continue
    
    _tasks[0].data[-1]['step'] = step
    _tasks[0].data[-1]['state'] = "RECV" if env_states[0] == STATE_RECV else "PROC"

    # Compute target transformation for RMPFlow Control
    target_transf = hand_transf.clone()
    marker_dspl_r, marker_dspl_transf, marker_dspl_dist = obs['marker_dspl_r'], obs['marker_dspl_transf'], obs['marker_dspl_dist']
    
    ## Proceed targets
    if len(proc_idx) > 0:
        curr_proceed_base_transf[proc_idx, :3, 3] = curr_proceed_base_transf[proc_idx, :3, 3] + curr_proceed_dir[proc_idx] * 0.0001
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
            ik_r = R.from_matrix(ik_r).as_quat()#[[3, 0, 1, 2]]
            if args.report:
                _tasks[i].report(False, False, 1e6)
            simulation_app.close()
            exit()
    
    # Update env states
    next_recv = torch.where(proc_flag * (marker_dspl_dist > delta_0))[0] # if step% 10 == 0 else torch.tensor([], dtype=torch.long, device='cuda')
    next_proc = torch.where(recv_flag * (marker_dspl_dist < delta_0 * alpha))[0] # if step% 10 == 0 else torch.tensor([], dtype=torch.long, device='cuda')
    next_succ_idx = torch.where(obs['achieved'] == 1.0)[0]
    
    env_states[next_proc] = STATE_PROC
    env_states[next_recv] = STATE_RECV
    env_states[next_succ_idx] = STATE_SUCC # Must come last to override.
    
    for i_proc in next_proc:
        _tasks[i_proc].attempt_counter += 1
    
    lock_joint_drive(lock_idx=next_recv.cpu().numpy().tolist())
    release_joint_drive(release_idx=next_proc.cpu().numpy().tolist())
    
    # Apply RMPFlow control
    target_pos, target_rot = target_transf[:, :3, 3].cpu().numpy(), target_transf[:, :3, :3].cpu().numpy()
    target_rot = R.from_matrix(target_rot).as_quat()
    rmpflow_action(target_pos, target_rot)

    if step % 100 == 99:
        tqdm.write(f"Env 0: Handle-to-target distance: {_tasks[0].to_target_dist} (succ: <= {SUCC_HANDLE_RANGE})")
    
    world.step(render=args.capture or (not args.headless))
    
simulation_app.close()
    
print(f"Bye. See results in {render_path}")
