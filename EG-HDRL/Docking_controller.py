#!/usr/bin/env python3
# coding: utf-8
"""
docking_controller.py – Visual-Servo Proportional-Depth Docking Controller
=========================================================================

Ensure the output action always stays within the SAC action space:
vx, vy, vz ∈ [-0.15, 0.15] m/s  wz ∈ [-0.10, 0.10] rad/s
"""

from __future__ import annotations
import numpy as np
import math
from typing import Tuple


class VSPDCController:
    """Visual-Servo Proportional-Depth Controller (Expert)"""

    def __init__(
        self,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        d_ref: float = 1.0,
        d_switch: float = 1.3,
        # Proportional gain for far/near distance
        kp_far_xy: float = 0.4,
        kp_near_xy: float = 0.15,
        kp_far_z: float = 0.45,
        kp_near_z: float = 0.18,
        kp_yaw: float = 0.6,           # rad/s per rad
        # Integral parameters
        ki_xy: float = 0.05,
        i_sat: float = 0.03,
        # Low-pass filter coefficient
        alpha: float = 0.85,
        # Far-field action limits consistent with SAC action space (vx, vy, vz, wz)
        v_limits_far: Tuple[float, float, float, float] = (0.15, 0.15, 0.15, 0.10),
        normalize: bool = False,
    ) -> None:
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.d_ref, self.d_switch = d_ref, d_switch
        self.kp_far_xy, self.kp_near_xy = kp_far_xy, kp_near_xy
        self.kp_far_z, self.kp_near_z = kp_far_z, kp_near_z
        self.kp_yaw = kp_yaw
        self.ki_xy, self.i_sat = ki_xy, i_sat
        self.alpha = alpha
        self.v_lim_far = v_limits_far          # Consistent with SAC action space
        self.normalize = normalize

        self.I_x = 0.0
        self.I_y = 0.0
        self.prev_cmd = np.zeros(4, dtype=np.float32)

    # ---------------------------- helper ---------------------------- #
    @staticmethod
    def _interp_gain(k_far: float, k_near: float, d: float, d_ref: float, d_sw: float) -> float:
        if d >= d_sw:
            return k_far
        if d <= d_ref:
            return k_near
        ratio = (d - d_ref) / (d_sw - d_ref)
        return k_near + (k_far - k_near) * ratio

    # ---------------------------------------------------------------- #
    def compute_action(self, obs: Tuple[float, float, float, float]) -> np.ndarray:
        u_or_x, v_or_y, d, theta_deg = obs

        # ---- 1. Pixel error → body error ---- #
        if self.normalize:
            ex_B = u_or_x * d
            ey_B = v_or_y * d
        else:
            du = self.cx - u_or_x
            dv = self.cy - v_or_y
            ex_cam = d / self.fx * du   # cam-x (right)
            ey_cam = d / self.fy * dv   # cam-y (down)
            ex_B = ey_cam               # body-forward
            ey_B = ex_cam               # body-right

        e_d = d - self.d_ref

        # ---- 2. Angular error (target is 90°) ---- #
        e_yaw = math.radians(theta_deg - 90.0)

        # ---- 3. P + I ---- #
        kp_xy = self._interp_gain(self.kp_far_xy, self.kp_near_xy, d, self.d_ref, self.d_switch)
        kp_z  = self._interp_gain(self.kp_far_z,  self.kp_near_z,  d, self.d_ref, self.d_switch)

        vx_p = kp_xy * ex_B
        vy_p = kp_xy * ey_B
        vz_p = -kp_z * e_d
        wz_p = -self.kp_yaw * e_yaw

        if abs(ex_B) < 0.05 and abs(ey_B) < 0.05:
            self.I_x = np.clip(self.I_x + ex_B * self.ki_xy, -self.i_sat, self.i_sat)
            self.I_y = np.clip(self.I_y + ey_B * self.ki_xy, -self.i_sat, self.i_sat)
        else:
            self.I_x = self.I_y = 0.0

        vx_i, vy_i = self.I_x, self.I_y
        cmd_raw = np.array([vx_p + vx_i, vy_p + vy_i, vz_p, wz_p], dtype=np.float32)

        # ---- 4. Low-pass filter + soft braking limits ---- #
        cmd_lp = self.alpha * self.prev_cmd + (1.0 - self.alpha) * cmd_raw

        # Distance coefficient 0 (near) → 1 (far)
        coef = np.clip((d - self.d_ref) / (self.d_switch - self.d_ref), 0.0, 1.0)

        # Near-field limits (consistent with drone_env success thresholds)
        v_xy_near, v_z_near, w_near = 0.05, 0.05, 0.02
        v_xy_lim = v_xy_near + (self.v_lim_far[0] - v_xy_near) * coef   # 0.05 → 0.15
        v_z_lim  = v_z_near  + (self.v_lim_far[2] - v_z_near)  * coef   # 0.05 → 0.15
        w_lim    = w_near    + (self.v_lim_far[3] - w_near)    * coef   # 0.02 → 0.10

        cmd_sat = np.clip(
            cmd_lp,
            [-v_xy_lim, -v_xy_lim, -v_z_lim, -w_lim],
            [ v_xy_lim,  v_xy_lim,  v_z_lim,  w_lim]
        )

        self.prev_cmd = cmd_sat
        return cmd_sat


# ------------------------------------------------------------------- #
if __name__ == "__main__":
    ctrl = VSPDCController(fx=454.6858, fy=454.6858, cx=424.5, cy=424.5)
    test = ctrl.compute_action((424.5 + 50, 424.5 + 30, 1.2, 5.0))
    print("Expert action:", test)
