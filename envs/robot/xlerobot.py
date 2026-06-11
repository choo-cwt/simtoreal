import copy
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


@register_agent()
class XLeRobot(BaseAgent):
    uid = "xlerobot_right_head"

    # Single right arm + head camera model from the repository-level urdf folder.
    # The head pan/tilt joints are fixed in the ManiSkill copy so the policy
    # keeps the same 6D single-arm action surface as the SO-101 experiments.
    urdf_path = str(Path(__file__).parent / "xlerobot_right_arm_head_fixed.urdf")

    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw_2=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            Moving_Jaw_2=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, -1.5708, 1.5708, 0.66, -np.pi, -10 * np.pi / 180]),
            pose=sapien.Pose(q=list(euler2quat(0, 0, 0))),
        ),
        start=Keyframe(
            qpos=np.array([0, 0, 0, np.pi / 2, np.pi / 2, 60 * np.pi / 180]),
            pose=sapien.Pose(q=list(euler2quat(0, 0, 0))),
        ),
        zero=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose(q=list(euler2quat(0, 0, 0))),
        ),
    )

    arm_joint_names = [
        "Rotation_R",
        "Pitch_R",
        "Elbow_R",
        "Wrist_Pitch_R",
        "Wrist_Roll_R",
    ]
    gripper_joint_names = ["Jaw_R"]
    gripper_joint_name = "Jaw_R"

    @property
    def _controller_configs(self):
        joint_names = self.arm_joint_names + self.gripper_joint_names

        pd_joint_pos = PDJointPosControllerConfig(
            joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )

        delta_lower = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.2]
        delta_upper = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2]

        pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names,
            delta_lower,
            delta_upper,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        pd_joint_vel = PDJointVelControllerConfig(
            joint_names,
            lower=[-1.0, -1.0, -1.0, -1.0, -1.0, -5.0],
            upper=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0],
            damping=[1e2] * 6,
            force_limit=100,
            friction=0,
            normalize_action=True,
        )

        return deepcopy_dict(
            dict(
                pd_joint_delta_pos=pd_joint_delta_pos,
                pd_joint_pos=pd_joint_pos,
                pd_joint_target_delta_pos=pd_joint_target_delta_pos,
                pd_joint_vel=pd_joint_vel,
            )
        )

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["Fixed_Jaw_2"]
        self.finger2_link = self.robot.links_map["Moving_Jaw_2"]

    def _link_local_point(self, link, local_xyz):
        mat = link.pose.to_transformation_matrix()
        local = torch.tensor(local_xyz, device=mat.device, dtype=mat.dtype)
        local = local.expand(*mat.shape[:-2], 3)
        ones = torch.ones((*mat.shape[:-2], 1), device=mat.device, dtype=mat.dtype)
        local_h = torch.cat([local, ones], dim=-1)
        return torch.matmul(mat, local_h.unsqueeze(-1)).squeeze(-1)[..., :3]

    @property
    def tcp_pos(self):
        fixed_tip = self._link_local_point(self.finger1_link, [0.0, -0.075, 0.0])
        moving_tip = self._link_local_point(self.finger2_link, [0.0, -0.075, 0.0])
        return (fixed_tip + moving_tip) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_touching(self, object: Actor):
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        return torch.logical_or(lforce >= 1e-2, rforce >= 1e-2)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.15):
        qvel = self.robot.get_qvel()[:, :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
