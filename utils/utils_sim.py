from collections import defaultdict

import carb
import torch

from omni.isaac.urdf import _urdf # Isaac Sim 2022
# from omni.importer.urdf import _urdf # Isaac Sim 2023, where some bugs exist
from PIL import Image

def get_default_import_config():
    # Load Object URDF
    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = True
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 50.0
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0
    
    return import_config


def init_capture():
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", True)
    
def write_rgb_data(rgb_data, file_path):
    rgb_img = Image.fromarray(rgb_data, "RGBA")
    rgb_img.save(file_path + ".png")
    

def fetch_obs(world, n_tasks):
    world_obs = world.get_observations()
    obs = defaultdict(list)
    
    for i in range(n_tasks):
        if f"{i}_obs" not in world_obs:
            continue
        
        for k, v in world_obs[f"{i}_obs"].items():
            obs[k].append(v)
                
    return { k: torch.stack(v, dim=0) for k, v in obs.items() }