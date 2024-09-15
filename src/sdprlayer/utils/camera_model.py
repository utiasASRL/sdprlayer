import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraModel(nn.Module):
    def __init__(
        self, f_u, f_v, gamma, c_u, c_v, sigma_u=0.0, sigma_v=0.0, check=False
    ):
        super(CameraModel, self).__init__()

        self.f_u = f_u
        self.f_v = f_v
        self.c_u = c_u
        self.c_v = c_v
        self.gamma = gamma
        self.sigma_u = torch.tensor(sigma_u)
        self.sigma_v = torch.tensor(sigma_v)

        # Define intrinsic
        self.K = torch.tensor(
            [
                [self.f_u, self.gamma, self.c_u],
                [0.0, self.f_v, self.c_v],
                [0.0, 0.0, 1.0],
            ]
        )
        # Check parameter
        self.check_valid = check

    def camera_to_image(self, cam_coords):
        """
        Project 3D points given in the camera frame into 2D image coordinates.

        Args:
            cam_coords (torch.tensor): 3D camera coordinates given as homogeneous coordinates, (Bx4xN).
            M (torch.tensor): matrix for projecting points into image, (Bx4x4).

        Returns:
            img_coords (torch.tensor): 2D image coordinates given in order (ul, vl, ur, vr), (Bx4xN).
        """
        batch_size, _, num_points = cam_coords.size()

        # [Ul, Vl, 1] = M * [x, y, z, 1]^T
        Ks = self.K.expand(cam_coords.size(0), 3, 3)
        img_coords = Ks.bmm(cam_coords[:, :3, :])

        inv_z = 1.0 / (
            cam_coords[:, 2, :].reshape(batch_size, 1, num_points)
        )  # Bx1xN, elementwise division.
        img_coords = img_coords * inv_z  # Bx4xN, elementwise multiplication.

        return img_coords

    def camera_model(self, cam_coords):
        """
        Project 3D points given in the camera frame into image coordinates.

        Args:
            cam_coords (torch.tensor): 3D camera coordinates given as homogeneous coordinates, (Bx4xN).

        Returns:
            img_coords (torch.tensor): 2D image coordinates given in order (ul, vl,1), (Bx3xN).
        """
        # Get the image coordinates.
        img_coords = self.camera_to_image(cam_coords)

        if self.check_valid:
            # Check validity of camera coordinates
            valid_img = torch.logical_not(
                torch.logical_or(torch.isnan(img_coords), torch.isinf(img_coords))
            )

            if not torch.all(valid_img):
                print("Warning: Nan or Inf values in image coordinate tensor.")
                # raise ValueError("Nan or Inf in image coordinates")

        return img_coords

    def normalize_coords(self, coords_2D, batch_size, height, width):
        """
        Normalize 2D keypoint coordinates to lie in the range [-1, 1].

        Args:
            coords_2D (torch.tensor): 2D image coordinates store in order (u, v), (Bx2xN).
            batch_size (int): batch size.
            height (int): image height.
            width (int): image width.

        Returns:
            coords_2D_norm (torch.tensor): coordinates normalized to range [-1, 1] and stored in order (u, v), (BxNx2).
        """
        u_norm = (2 * coords_2D[:, 0, :].reshape(batch_size, -1) / (width - 1)) - 1
        v_norm = (2 * coords_2D[:, 1, :].reshape(batch_size, -1) / (height - 1)) - 1

        return torch.stack([u_norm, v_norm], dim=2)
