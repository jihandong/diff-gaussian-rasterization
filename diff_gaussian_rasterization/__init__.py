#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            getattr(raster_settings, 'profile_mask', 0)
        )

        # Invoke C++/CUDA rasterizer
        # HJ-Profiling: The C++ extension now optionally returns two additional
        # profiling tensors (tests_per_pixel, contribs_per_pixel) at the
        # end of the returned tuple. We always unpack them, but only
        # return them to the caller when debug is enabled in
        # raster_settings.
        result = _C.rasterize_gaussians(*args)
        # Supported return arities from native (after adding final_T):
        # 8  -> base (no profiling)
        # 12 -> counts only (tests, contribs, first_true_at, post_false_after)
        # 14 -> counts + timing (adds loop_cycles, discrim_cycles)
        if isinstance(result, (list, tuple)):
            if len(result) == 8:
                (num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer,
                 invdepths, final_T) = result
                tests_per_pixel = torch.empty(0, dtype=torch.int32, device=color.device)
                contribs_per_pixel = torch.empty(0, dtype=torch.int32, device=color.device)
                first_true_at = torch.empty(0, dtype=torch.int32, device=color.device)
                post_false_after = torch.empty(0, dtype=torch.int32, device=color.device)
                loop_cycles = torch.empty(0, dtype=torch.int64, device=color.device)
                discrim_cycles = torch.empty(0, dtype=torch.int64, device=color.device)
            elif len(result) == 12:
                (num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, final_T,
                 tests_per_pixel, contribs_per_pixel, first_true_at, post_false_after) = result
                loop_cycles = torch.empty(0, dtype=torch.int64, device=color.device)
                discrim_cycles = torch.empty(0, dtype=torch.int64, device=color.device)
            elif len(result) == 14:
                (num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, final_T,
                 tests_per_pixel, contribs_per_pixel, first_true_at, post_false_after, loop_cycles, discrim_cycles) = result
            else:
                raise RuntimeError(f"Unexpected rasterizer return arity: {len(result)}")
        else:
            raise RuntimeError("Rasterizer returned non-tuple result")

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        # Always return a fixed 10-tuple to the renderer, filling empties as needed.
        # (color, final_T, radii, invdepths, tests, contribs, first_true, post_false, loop_cycles, discrim_cycles)
        tests_out = tests_per_pixel if tests_per_pixel.numel() > 0 else torch.empty(0, dtype=torch.int32, device=color.device)
        contribs_out = contribs_per_pixel if contribs_per_pixel.numel() > 0 else torch.empty(0, dtype=torch.int32, device=color.device)
        first_true_out = first_true_at if first_true_at.numel() > 0 else torch.empty(0, dtype=torch.int32, device=color.device)
        post_false_out = post_false_after if post_false_after.numel() > 0 else torch.empty(0, dtype=torch.int32, device=color.device)
        loop_out = loop_cycles if loop_cycles.numel() > 0 else torch.empty(0, dtype=torch.int64, device=color.device)
        discrim_out = discrim_cycles if discrim_cycles.numel() > 0 else torch.empty(0, dtype=torch.int64, device=color.device)
        final_T_out = final_T if final_T.numel() > 0 else torch.empty(0, dtype=color.dtype, device=color.device)
        return (
            color, final_T_out, radii, invdepths,
            tests_out, contribs_out, first_true_out, post_false_out,
            loop_out, discrim_out
        )

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            raster_settings.profile_mask)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    profile_mask : int

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

