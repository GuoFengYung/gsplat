"""Python bindings for 3D gaussian projection"""

from typing import Optional, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import gsplat.cuda as _C


def project_gaussians(
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"
    return _ProjectGaussians.apply(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        viewmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )


class _ProjectGaussians(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
            ctx,
            means3d: Float[Tensor, "*batch 3"],
            scales: Float[Tensor, "*batch 3"],
            glob_scale: float,
            quats: Float[Tensor, "*batch 4"],
            viewmat: Float[Tensor, "4 4"],
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            img_height: int,
            img_width: int,
            block_width: int,
            clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        )

        return (xys, depths, radii, conics, compensation, num_tiles_hit, cov3d)

    @staticmethod
    def backward(
            ctx,
            v_xys,
            v_depths,
            v_radii,
            v_conics,
            v_compensation,
            v_num_tiles_hit,
            v_cov3d,
    ):
        (
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        ) = ctx.saved_tensors

        (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat) = _C.project_gaussians_backward(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            compensation,
            v_xys,
            v_depths,
            v_conics,
            v_compensation,
        )

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Denote ProjectGaussians for a single Gaussian (mean3d, q, s)
            # viemwat = [R, t] as:
            #
            #   f(mean3d, q, s, R, t, intrinsics)
            #       = g(R @ mean3d + t,
            #           R @ cov3d_world(q, s) @ R^T ))
            #
            # Then, the Jacobian w.r.t., t is:
            #
            #   d f / d t = df / d mean3d @ R^T
            #
            # and, in the context of fine tuning camera poses, it is reasonable
            # to assume that
            #
            #   d f / d R_ij =~ \sum_l d f / d t_l * d (R @ mean3d)_l / d R_ij
            #                = d f / d_t_i * mean3d[j]
            #
            # Gradients for R and t can then be obtained by summing over
            # all the Gaussians.
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(
                        v_mean3d_cam[..., j], means3d[..., l]
                    )
        else:
            v_viewmat = None

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # scales: Float[Tensor, "*batch 3"],
            v_scale,
            # glob_scale: float,
            None,
            # quats: Float[Tensor, "*batch 4"],
            v_quat,
            # viewmat: Float[Tensor, "4 4"],
            v_viewmat,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # block_width: int,
            None,
            # clip_thresh,
            None,
        )


def project_gaussians_2d(
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        opacity: Float[Tensor, "*batch 1"],
        viewmat: Float[Tensor, "4 4"],
        projmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projwmat (Tensor): proj matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **transMats** (Tensor):
        - **normal** (Tensor):
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    # assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"
    return _ProjectGaussians_2d.apply(
        means3d.contiguous(),
        scales.contiguous(),
        glob_scale,
        quats.contiguous(),
        opacity.contiguous(),
        viewmat.contiguous(),
        projmat.contiguous(),
        fx,
        fy,
        cx,
        cy,
        img_height,
        img_width,
        block_width,
        clip_thresh,
    )


class _ProjectGaussians_2d(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
            ctx,
            means3d: Float[Tensor, "*batch 3"],
            scales: Float[Tensor, "*batch 3"],
            glob_scale: float,
            quats: Float[Tensor, "*batch 4"],
            opacity: Float[Tensor, "*batch 1"],
            viewmat: Float[Tensor, "4 4"],
            projwmat: Float[Tensor, "4 4"],
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            img_height: int,
            img_width: int,
            block_width: int,
            clip_thresh: float = 0.01
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            xys,
            depths,
            transMats,
            normal,
            radii,
            num_tiles_hit,
        ) = _C.project_gaussians_forward_2d(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            opacity,
            viewmat,
            projwmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy
        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            transMats,
            projwmat,
            radii,
            normal,
        )

        return xys, depths, transMats, normal, radii, num_tiles_hit
    @staticmethod
    def backward(ctx,
                 xys,
                 depths,
                 transMats,
                 normal,
                 radii,
                 num_tiles_hit):
        (
            means3d,
            scales,
            quats,
            viewmat,
            transMats,
            projwmat,
            radii,
            normal,
        ) = ctx.saved_tensors
        # cuda
        # print("means3d shape:", means3d.shape, "type:", type(means3d), "dtype:", means3d.dtype)
        # print("transMats shape:", transMats.shape, "type:", type(transMats), "dtype:", transMats.dtype)
        # print("scales shape:", scales.shape, "type:", type(scales), "dtype:", scales.dtype)
        # print("quats shape:", quats.shape, "type:", type(quats), "dtype:", quats.dtype)
        # print("viewmat shape:", viewmat.shape, "type:", type(viewmat), "dtype:", viewmat.dtype)
        # print("projwmat shape:", projwmat.shape, "type:", type(projwmat), "dtype:", projwmat.dtype)
        # print("radii shape:", radii.shape, "type:", type(radii), "dtype:", radii.dtype)
        # print("dL_dnormal3Ds shape:", dL_dnormal3Ds.shape, "type:", type(dL_dnormal3Ds), "dtype:", dL_dnormal3Ds.dtype)
        (dL_dtransMats,
         dL_dmean2Ds,
         dL_dmean3Ds,
         dL_dscales,
         dL_drots) = _C.project_gaussians_backward_2d(
            ctx.num_points,
            means3d,
            transMats,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            projwmat,
            ctx.img_height,
            ctx.img_width,
            radii,
            normal
        )
        # debug
        print("project bp1")
        print("Shape and type of dL_dtransMats:", dL_dtransMats.shape, type(dL_dtransMats))
        print("Shape and type of dL_dmean2Ds:", dL_dmean2Ds.shape, type(dL_dmean2Ds))
        print("Shape and type of dL_dmean3Ds:", dL_dmean3Ds.shape, type(dL_dmean3Ds))
        print("Shape and type of dL_dscales:", dL_dscales.shape, type(dL_dscales))
        print("Shape and type of dL_drots:", dL_drots.shape, type(dL_drots))

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Denote ProjectGaussians for a single Gaussian (mean3d, q, s)
            # viemwat = [R, t] as:
            #
            #   f(mean3d, q, s, R, t, intrinsics)
            #       = g(R @ mean3d + t,
            #           R @ cov3d_world(q, s) @ R^T ))
            #
            # Then, the Jacobian w.r.t., t is:
            #
            #   d f / d t = df / d mean3d @ R^T
            #
            # and, in the context of fine tuning camera poses, it is reasonable
            # to assume that
            #
            #   d f / d R_ij =~ \sum_l d f / d t_l * d (R @ mean3d)_l / d R_ij
            #                = d f / d_t_i * mean3d[j]
            #
            # Gradients for R and t can then be obtained by summing over
            # all the Gaussians.
            v_mean3d_cam = torch.matmul(dL_dmean3Ds, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(
                        v_mean3d_cam[..., j], means3d[..., l]
                    )
        else:
            v_viewmat = None

        dL_dscales = torch.concatenate((dL_dscales, torch.zeros((dL_dscales.shape[0], 1), device="cuda:0")), dim=-1)
        # pdb.set_trace()
        # Return a gradient for each input.
        # print(v_mean3d)
        # print(v_scale)
        # print(v_quat)
        return (
            # means3d: Float[Tensor, "*batch 3"],
            dL_dmean3Ds,
            # scales: Float[Tensor, "*batch 3"],
            dL_dscales,
            # glob_scale: float,
            None,
            # quats: Float[Tensor, "*batch 4"],
            dL_drots,
            # normal
            None,
            # viewmat: Float[Tensor, "4 4"],
            v_viewmat,
            # proj
            None,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # block_width: int,
            None,
            # clip_thresh,
            None
        )
