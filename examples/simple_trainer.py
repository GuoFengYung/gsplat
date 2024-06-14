import math
import os
import time
from pathlib import Path
from typing import Optional
import matplotlib
import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians, project_gaussians_2d
from gsplat.rasterize import rasterize_gaussians, rasterize_gaussians_2d
from PIL import Image
from torch import Tensor, optim


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
            self,
            gt_image: Tensor,
            num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points
        self.znear = 0.01
        self.zfar = 100
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    #
    # def focal2fov(self, focal, pixels):
    #     return 2 * math.atan(pixels / (2 * focal))
    #
    # def get_cameras(self):
    #     intrins = torch.tensor([[711.1111, 0.0000, 256.0000, 0.0000],
    #                             [0.0000, 711.1111, 256.0000, 0.0000],
    #                             [0.0000, 0.0000, 1.0000, 0.0000],
    #                             [0.0000, 0.0000, 0.0000, 1.0000]]).cuda()
    #
    #     c2w = torch.tensor([[-8.6086e-01, 3.7950e-01, -3.3896e-01, 6.7791e-01],
    #                         [5.0884e-01, 6.4205e-01, -5.7346e-01, 1.1469e+00],
    #                         [1.0934e-08, -6.6614e-01, -7.4583e-01, 1.4917e+00],
    #                         [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]).cuda()
    #
    #     width, height = 512, 512
    #     focal_x, focal_y = intrins[0, 0], intrins[1, 1]
    #     viewmat = torch.linalg.inv(c2w).permute(1, 0)
    #     FoVx = self.focal2fov(focal_x, width)
    #     FoVy = self.focal2fov(focal_y, height)
    #     projmat = self.getProjectionMatrix(znear=0.2, zfar=1000, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
    #     projmat = viewmat @ projmat
    #     return intrins, viewmat, projmat, height, width

    # def get_inputs(self, num_points=8):
    #     length = 0.5

        # x = np.linspace(-1, 1, num_points) * length
        # y = np.linspace(-1, 1, num_points) * length
        # x, y = np.meshgrid(x, y)

        # means3D = torch.from_numpy(
        #     np.stack([x, y, 0 * np.random.rand(*x.shape)], axis=-1).reshape(-1, 3)
        # ).cuda().float()

        # quats = torch.zeros(1, 4).repeat(len(means3D), 1).cuda()
        # quats[..., 0] = 1.
        # quats = quats / quats.norm(dim=-1, keepdim=True)
        # u = torch.rand(len(means3D), 1, device=self.device)
        # v = torch.rand(len(means3D), 1, device=self.device)
        # w = torch.rand(len(means3D), 1, device=self.device)
        #
        # quats = torch.cat(
        #     [
        #         torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        #         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        #         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        #         torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        #     ],
        #     -1,
        # )

        # scale = length / (num_points - 1)
        # scales = torch.zeros(1, 3).repeat(len(means3D), 1).fill_(scale).cuda()

        # opacities = torch.ones((len(means3D), 1)).cuda()

        # def fixed_colors(means3D, colormap='Accent'):
        #     colormap = matplotlib.colormaps[colormap]
        #     num_points = len(means3D)
        #     indices = np.arange(num_points)
        #     colors = colormap(indices / num_points)[..., :3]  # 取前3个通道作为RGB
        #     return colors

        # colors = fixed_colors(means3D)

        # colors = torch.from_numpy(colors).to(torch.float).cuda()

        # return means3D, scales, quats, opacities, colors

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        self.projmat = self.getProjectionMatrix(znear=0.2, zfar=1000, fovX=self.focal, fovY=self.focal).transpose(0, 1).cuda()

        d = 3
        self.background = torch.zeros(d, device=self.device)
        # self.means3D, self.scales, self.quats, self.opacities, self.colors = self.get_inputs(num_points=bd)
        # intrins, self.viewmat, self.projmat, self.height, self.width = self.get_cameras()
        # intrins = intrins[:3, :3]
        # self.focal_x, self.focal_y = intrins[0, 0], intrins[1, 1]
        # self.cx, self.cy = intrins[0, 2], intrins[1, 2]
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)
        # self.projmat = getProjectionMatrix(self.znear, self.zfar, fov_x, fov_y)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        # self.means3D.requires_grad = True
        # self.scales.requires_grad = True
        # self.quats.requires_grad = True
        # self.colors.requires_grad = True
        # self.opacities.requires_grad = True
        # self.viewmat.requires_grad = False

    def train(
            self,
            iterations: int = 1000,
            lr: float = 0.01,
            save_imgs: bool = False,
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        # optimizer = optim.Adam(
        #     [self.means3D, self.scales, self.quats, self.opacities, self.colors], lr
        # )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        B_SIZE = 16
        for iter in range(iterations):
            start = time.time()
            # print("view shape:", self.viewmat.shape)
            xys, depths, transMats, normal, radii, num_tiles_hit = project_gaussians_2d(
                self.means,
                self.scales,
                1,
                self.quats,
                self.opacities,
                self.viewmat,
                self.projmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                B_SIZE,
                0.01
            )

            torch.cuda.synchronize()

            times[0] += time.time() - start
            start = time.time()
            # print(normal[:, 3:4].shape)
            out_img = rasterize_gaussians_2d(xys,
                                                transMats,
                                                depths,
                                                radii,
                                                num_tiles_hit,
                                                torch.sigmoid(self.rgbs),
                                                torch.sigmoid(normal),
                                                self.H,
                                                self.W,
                                                B_SIZE,
                                                background=torch.zeros(3).cuda(),
                                                return_alpha=False)

            # img1 = out_img.detach().cpu().numpy()
            # rgb_np = (img1 * 255).astype(np.uint8)
            # image = Image.fromarray(rgb_np)
            # image.save('output_image.jpg')

            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0] / iterations:.5f}, Rasterize: {times[1] / iterations:.5f}, Backward: {times[2] / iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
        height: int = 512,
        width: int = 512,
        num_points: int = 500,
        save_imgs: bool = True,
        img_path: Optional[Path] = None,
        iterations: int = 1000,
        lr: float = 0.01,
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2:, width // 2:, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)
