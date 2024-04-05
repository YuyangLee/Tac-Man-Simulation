import importlib
import json
import os
from itertools import permutations

import numpy as np
import plotly.graph_objects as go
import pytorch_kinematics as pk
import torch
import transforms3d
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from loguru import logger
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh,
                                                    Sphere)

from utils.utils_3d import compute_rotation_matrix_from_ortho6d

try:
    importlib.find_loader('kaolin')
    from kaolin.metrics.trianglemesh import point_to_mesh_distance
    from kaolin.ops.mesh import (check_sign, face_normals,
                                 index_vertices_by_faces)
    logger.info(f"Successfully loaded kaolin")
except ImportError:
    logger.warning(f"Failed to load kaolin. Functions like ODF training are disabled.")
    logger.warning(f"Failed to load kaolin. Functions like object-centric penetration and ODF training are disabled.")
        

class ArticulatedObj:
    def __init__(self, obj_model, urdf_filename, mesh_path, specs_path=None,
                 batch_size=1, obj_scale=1., pts_density=3,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 load_kaolin=False,
                 **kwargs
                 ):
        self.device = device
        self.obj_model = obj_model
        self.batch_size = batch_size
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)

        self.joint_param_names = self.robot.get_joint_parameter_names()
        self.q_len = 9
        
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)
        self.surface_points = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}
        self.num_links = len(visual.links)
        
        self.num_pts = 0
        
        if load_kaolin:                           
            self.canon_verts = []
            self.canon_faces = []
            self.idx_vert_faces = []
            self.face_normals = []

        for i_link, link in enumerate(visual.links):
            if len(link.visuals) == 0:
                continue
            
            self.surface_points[link.name] = []
            self.mesh_verts[link.name] = []
            self.mesh_faces[link.name] = []
            
            for link_viz in link.visuals:
                if type(link_viz.geometry) == Mesh:
                    if "//" in link_viz.geometry.filename:
                        filename = link_viz.geometry.filename.split('/')[-1]
                    else:
                        filename = link_viz.geometry.filename
                        
                    mesh = tm.load(os.path.join(mesh_path, filename), force='mesh', process=False)
                elif type(link_viz.geometry) == Cylinder:
                    mesh = tm.primitives.Cylinder(
                        radius=link_viz.geometry.radius, height=link_viz.geometry.length)
                elif type(link_viz.geometry) == Box:
                    mesh = tm.primitives.Box(extents=link_viz.geometry.size)
                elif type(link_viz.geometry) == Sphere:
                    mesh = tm.primitives.Sphere(
                        radius=link_viz.geometry.radius)
                else:
                    raise NotImplementedError
                try:
                    scale = np.array(link_viz.geometry.scale).reshape([1, 3])
                except:
                    scale = np.array([[1, 1, 1]])
                    
                try:
                    rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                    translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
                except AttributeError:
                    rotation = transforms3d.euler.euler2mat(0, 0, 0)
                    translation = np.array([[0, 0, 0]])
                    
                num_part_pts = max(20, int(mesh.area * pts_density * 10000))
                pts = mesh.sample(num_part_pts) * scale
                self.num_pts += num_part_pts

                # Surface Points
                pts = np.matmul(rotation, pts.T).T + translation
                pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
                self.surface_points[link.name].append(torch.from_numpy(pts).to(device).float().unsqueeze(0).repeat(batch_size, 1, 1))

                # Visualization Mesh
                self.mesh_verts[link.name].append(np.matmul(rotation, (np.array(mesh.vertices) * scale).T).T + translation)
                self.mesh_faces[link.name].append(np.array(mesh.faces) + sum([len(x) for x in self.mesh_verts[link.name][:-1]]))
            
            self.surface_points[link.name] = torch.cat(self.surface_points[link.name], dim=1)    
            self.mesh_verts[link.name] = np.concatenate(self.mesh_verts[link.name], axis=0)
            self.mesh_faces[link.name] = np.concatenate(self.mesh_faces[link.name], axis=0)
                        
        # self.revolute_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type in [ 'revolute', 'continuous', 'prismatic' ] :
                self.q_len += 1
                # self.revolute_joints.append(self.robot_full.joints[i])
        # self.revolute_joints_q_mid = []
        # self.revolute_joints_q_var = []
        # self.revolute_joints_q_upper = []
        # self.revolute_joints_q_lower = []
        # for i in range(len(self.joint_param_names)):
        #     for j in range(len(self.revolute_joints)):
        #         if self.revolute_joints[j].name == self.joint_param_names[i]:
        #             joint = self.revolute_joints[j]
        #     assert joint.name == self.joint_param_names[i]
        #     self.revolute_joints_q_mid.append((joint.limit.lower + joint.limit.upper) / 2)
        #     self.revolute_joints_q_var.append(((joint.limit.upper - joint.limit.lower) / 2) ** 2)
        #     self.revolute_joints_q_lower.append(joint.limit.lower)
        #     self.revolute_joints_q_upper.append(joint.limit.upper)

        # self.revolute_joints_q_lower = torch.Tensor(self.revolute_joints_q_lower).to(device)
        # self.revolute_joints_q_upper = torch.Tensor(self.revolute_joints_q_upper).to(device)

        self.current_status = None

        self.canon_pose = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0] + [0] * (self.q_len - 9), device=device, dtype=torch.float32)
        self.scale = obj_scale
        
        self.update_kinematics(self.canon_pose.unsqueeze(0))

    def random_q(self, batch_size, one_traj=True):
        transf = torch.normal(0, 1, [batch_size, 9], device=self.device, dtype=torch.float32)
        # joints = torch.rand([batch_size, self.q_len - 9], device=self.device, dtype=torch.float32)
        joints = torch.rand([batch_size, self.q_len - 9], device=self.device, dtype=torch.float32) * 0.5
        joints = joints * (self.revolute_joints_q_upper - self.revolute_joints_q_lower) + self.revolute_joints_q_lower
        q = torch.cat([transf, joints], dim=-1)
        
        q[:, 0:9] = 0.0
        if self.obj_model == "f_tac":
            q[:, 2] = 0.4
        
        R6 = torch.normal(0, 1, [batch_size, 6], dtype=torch.float32, device=self.device)
        R = compute_rotation_matrix_from_ortho6d(R6)
        
        q[:, 0:3] = torch.matmul(R, q[:, 0:3].unsqueeze(-1)).squeeze()
        q[:, 3:9] = R6.clone() + torch.normal(0, 0.1, [batch_size, 6], dtype=torch.float32, device=self.device)
        
        if one_traj:
            q = q.unsqueeze(1)
            
        q = q.contiguous().clone()
        q.requires_grad_()
        return q

    def update_kinematics(self, q):
        self.batch_size = q.shape[0]
        self.global_translation = q[:, :3] / self.scale
        self.global_rotation = compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])
            
    def penetration(self, obj_pts, q=None):
        """Penetration of object points in the conanical frame

        Args:
            q: B x l
            obj_pts: B x N x 4
        """
        if q is not None:
            self.update_kinematics(q)
        oh_pen = torch.zeros([1, obj_pts.shape[0] * obj_pts.shape[1]], device=self.device)

        # Transform point to the object frame
        local_obj_pts = torch.matmul(self.global_rotation.transpose(1, 2), (obj_pts[..., :3] - self.global_translation.unsqueeze(1) * self.scale).transpose(1, 2)).transpose(1, 2)
        pts_shape = obj_pts[..., :3].shape
        
        for link_idx, link_name in enumerate(self.surface_points):
            # Transform point to the canonical part frame
            trans_matrix = self.current_status[link_name].get_matrix()
            lp_obj_pts = (torch.matmul(trans_matrix[:, :3, :3].transpose(1, 2), (local_obj_pts.clone() - trans_matrix[:, :3, -1].unsqueeze(1) * self.scale).transpose(1, 2)).transpose(1, 2))
            _lp_obj_pts = lp_obj_pts.contiguous().reshape((1, -1, 3))
            # Compute penetration
            oh_dist, _, _ = point_to_mesh_distance(_lp_obj_pts, self.idx_vert_faces[link_idx])
            oh_sign = check_sign(self.canon_verts[link_idx], self.canon_faces[link_idx], _lp_obj_pts)
            oh_pen = oh_pen + torch.where(oh_sign, oh_dist, torch.zeros_like(oh_dist, device=self.device))
            
        return oh_pen.reshape((pts_shape[0], pts_shape[1]))
    
    def prior(self, q):
        range_energy = torch.relu(q[:, 9:] - self.revolute_joints_q_upper) + torch.relu(self.revolute_joints_q_lower - q[:, 9:])
        return range_energy.sum(-1)

    def get_pointcloud(self, q=None, links=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for i_link, link_name in enumerate(self.surface_points):
            if links is not None and link_name not in links:
                continue
            trans_matrix = self.current_status[link_name].get_matrix().expand([self.batch_size, 4, 4])
            pts = torch.matmul(trans_matrix, self.surface_points[link_name].transpose(1, 2)).transpose(1, 2)[..., :3]
            if i_link == 0:
               pts = pts.expand([self.batch_size, pts.shape[1], pts.shape[2]])
                
            surface_points.append(pts)
            
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale
    
    def get_meshes_from_q(self, q=None, i=0, links=None, concat=False):
        data = []
        if q is not None: self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            if links is not None and link_name not in links:
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(), transformed_v.T).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        if concat:
            data = tm.util.concatenate(data)
        return data
    
    def get_link_local_meshes_from_q(self, link):
        return tm.Trimesh(vertices=self.mesh_verts[link], faces=self.mesh_faces[link])
    