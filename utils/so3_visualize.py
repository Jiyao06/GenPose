import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch3d
import cv2

from matplotlib import rc
from PIL import Image
from pytorch3d import transforms
from ipdb import set_trace


rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
EYE = np.eye(3)

def visualize_so3_probabilities(
    rotations,
    probabilities,
    rotations_gt=None,
    choosed_rotation=None,
    ax=None,
    fig=None,
    display_threshold_probability=0,
    to_image=True,
    show_color_wheel=True,
    canonical_rotation=EYE,
    gt_size=600,
    choosed_size=300,
    y_offset=-30,
    dpi=600,
):
    """
    Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
        rotations: [N, 3, 3] tensor of rotation matrices
        probabilities: [N] tensor of probabilities
        rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
        ax: The matplotlib.pyplot.axis object to paint
        fig: The matplotlib.pyplot.figure object to paint
        display_threshold_probability: The probability threshold below which to omit
            the marker
        to_image: If True, return a tensor containing the pixels of the finished
            figure; if False return the figure itself
        show_color_wheel: If True, display the explanatory color wheel which matches
            color on the plot with tilt angle
        canonical_rotation: A [3, 3] rotation matrix representing the 'display
            rotation', to change the view of the distribution.  It rotates the
            canonical axes so that the view of SO(3) on the plot is different, which
            can help obtain a more informative view.
    Returns:
        A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """

    def _show_single_marker(ax, rotation, marker, edgecolors=True, facecolors=False, s=gt_size):
        eulers = transforms.matrix_to_euler_angles(torch.tensor(rotation), "ZXY")
        eulers = eulers.numpy()

        tilt_angle = eulers[0]
        latitude = eulers[1]
        longitude = eulers[2]

        color = cmap(0.5 + tilt_angle / 2 / np.pi)
        ax.scatter(
            longitude,
            latitude,
            s=s,
            edgecolors=color if edgecolors else "none",
            facecolors=facecolors if facecolors else "none",
            marker=marker,
            linewidth=5,
        )

    if ax is None:
        fig = plt.figure(figsize=(4, 2), dpi=dpi)
        ax = fig.add_subplot(111, projection="mollweide")
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
        rotations_gt = rotations_gt[None]
    if choosed_rotation is not None and len(choosed_rotation.shape) == 2:
        choosed_rotation = choosed_rotation[None]
    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e3
    eulers_queries = transforms.matrix_to_euler_angles(
        torch.tensor(display_rotations), "ZXY"
    )
    eulers_queries = eulers_queries.numpy()

    tilt_angles = eulers_queries[:, 0]
    longitudes = eulers_queries[:, 2]
    latitudes = eulers_queries[:, 1]

    which_to_display = probabilities > display_threshold_probability

    if rotations_gt is not None:
        display_rotations_gt = rotations_gt @ canonical_rotation

        for rotation in display_rotations_gt:
            _show_single_marker(ax, rotation, "o")
        # Cover up the centers with white markers
        for rotation in display_rotations_gt:
            _show_single_marker(
                ax, rotation, "o", edgecolors=False, facecolors="#ffffff"
            )

    if choosed_rotation is not None:
        display_choosed_rotations = choosed_rotation @ canonical_rotation

        for rotation in display_choosed_rotations:
            _show_single_marker(ax, rotation, "o", s=choosed_size)
        # Cover up the centers with white markers
        for rotation in display_choosed_rotations:
            _show_single_marker(
                ax, rotation, "o", edgecolors=False, facecolors="#ffffff", s=choosed_size
            )
            
    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2.0 / np.pi),
        marker='.'
    )

    yticks = np.array([-60, -30, 0, 30, 60])
    yticks_minor = np.arange(-75, 90, 15)
    ax.set_yticks(yticks_minor * np.pi / 180, minor=True)
    ax.set_yticks(yticks * np.pi / 180, [f"{y}°" for y in yticks], fontsize=10)
    xticks = np.array([-90, 0, 90])
    xticks_minor = np.arange(-150, 180, 30)
    ax.set_xticks(xticks * np.pi / 180, [])
    ax.set_xticks(xticks_minor * np.pi / 180, minor=True)

    for xtick in xticks:
        # Manually set xticks
        x = xtick * np.pi / 180
        y = y_offset * np.pi / 180
        ax.text(x, y, f"{xtick}°", ha="center", va="center", fontsize=10)

    ax.grid(which="minor")
    ax.grid(which="major")

    if show_color_wheel:
        # Add a color wheel showing the tilt angle to color conversion.
        ax = fig.add_axes([0.85, 0.12, 0.12, 0.12], projection="polar")
        theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
        radii = np.linspace(0.4, 0.5, 2)
        _, theta_grid = np.meshgrid(radii, theta)
        colormap_val = 0.5 + theta_grid / np.pi / 2.0
        ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap, shading="auto")
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 2))
        ax.set_xticklabels(
            [
                r"90$\degree$",
                r"180$\degree$",
                r"270$\degree$",
                r"0$\degree$",
            ],
            fontsize=6,
        )
        ax.spines["polar"].set_visible(False)
        ax.grid(False)
        plt.text(
            0.5,
            0.5,
            "Roll",
            fontsize=6,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    if to_image:
        return plot_to_image(fig)
    else:
        return fig


def plot_to_image(fig):
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    plt.close(fig)
    return image_from_plot


def antialias(image, level=1):
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        image = Image.fromarray(image)
    for _ in range(level):
        size = np.array(image.size) // 2
        image = image.resize(size, Image.LANCZOS)
    if is_numpy:
        image = np.array(image)
    return image


def unnormalize_image(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return (image * 255.0).astype(np.uint8)


def visualize_so3(save_path, pred_rotations, gt_rotation, pred_rotation=None, probabilities=None, image=None):
    if image == None:
        fig = plt.figure(figsize=(5, 2), dpi=600)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, :], projection="mollweide")
    else:
        fig = plt.figure(figsize=(5, 4), dpi=600)
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[1, :], projection="mollweide")  
        bx = fig.add_subplot(gs[0, :])  
        bx.imshow(image.permute(1, 2, 0).cpu().numpy())
        bx.axis("off")
    # rotations = np.concatenate((pred_rotations, pred_rotation, gt_rotation), axis=0)
    rotations = pred_rotations
    if probabilities is None:
        probabilities = np.ones(rotations.shape[0])/2000
    # probabilities[-2] = 0.002
    # probabilities[-1] = 0.003
    
    so3_vis = visualize_so3_probabilities(
        rotations_gt=gt_rotation,
        choosed_rotation=pred_rotation,
        rotations=rotations,
        probabilities=probabilities,
        to_image=False,
        display_threshold_probability=0.00001,
        show_color_wheel=True,
        fig=fig,
        ax=ax,
    )
    plt.savefig(save_path)
    
    
if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 4), dpi=300)
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    # ax1.imshow(unnormalize_image(images[0].cpu().numpy().transpose(1, 2, 0)))
    ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, 1])
    # ax2.imshow(unnormalize_image(images[1].cpu().numpy().transpose(1, 2, 0)))
    ax2.axis("off")
    ax3 = fig.add_subplot(gs[1, :], projection="mollweide")

    rotations = pytorch3d.transforms.random_rotations(10).cpu().numpy()
    probabilities = np.ones(10)/1000
    print(probabilities)
    so3_vis = visualize_so3_probabilities(
        rotations=rotations,
        probabilities=probabilities,
        to_image=False,
        display_threshold_probability=0.0005,
        show_color_wheel=True,
        fig=fig,
        ax=ax3,
    )
    plt.savefig('./test.jpg')
    # plt.show()

