#!/usr/bin/self python3
# -*- coding: utf-8 -*-
"""Provides the Visualization class, for rendering episodes."""
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Visualization:
    """ A class for visualizing the search in 1D, 2D or 3D

    Args:
        env (SourceTracking):
            an instance of the SourceTracking class
        live (bool, optional):
            whether to show live preview (faster if False) (default=False)
        filename (str, optional):
            file name for the video (default='test')
        log_prob (bool, optional):
            whether to show log(prob) instead of prob (default=False)
        marginal_prob_3d (bool, optional):
            in 3D, whether to show marginal pdfs on each plane, instead of the pdf in the planes that the
            agent crosses (default=False)
    """
    
    def __init__(self,
                 env,
                 live=False,
                 filename='test',
                 log_prob=False,
                 marginal_prob_3d=False,
                 ):
        self.env = env
        if self.env.Ndim > 3 or self.env.Ndim < 1 or not isinstance(env.Ndim, int):
            raise Exception("Problem with Ndim: visualization is not possible")

        self.video_live = live
        self.frame_path = filename + "_frames"
        self.video_path = filename + "_video"
        if not os.path.isdir(self.frame_path):
            os.mkdir(self.frame_path)

        self.log_prob = log_prob
        self.marginal_prob_3d = marginal_prob_3d

    def make_video(self, frame_rate=5, keep_frames=False):
        """
        Make a video from recorded frames and clean up frames.

        Args:
            frame_rate (int): number of frames per second (default=5)
            keep_frames (bool): whether to keep the frames as images (default=False)

        Returns:
            exit_code (int):
                nonzero if something went wrong while making the video, in that case frames will be
                saved even if keep_frames = False
        """
        if self.video_live:
            plt.close("all")

        exit_code = self._make_video(frame_rate=frame_rate, keep_frames=keep_frames)
        return exit_code

    def record_snapshot(self, num, toptext=''):
        """Create a frame from current state of the search, and save it.

        Args:
            num (int): frame number (used to create filename)
            toptext (str): text that will appear in the top part of the frame (like a title)
        """

        if self.video_live:
            if not hasattr(self, 'fig'):
                fig, ax = self._setup_render()
                ax[0].set_title("observation map (current: %s)" % self._obs_to_str())
                ax[1].set_title("source probability distribution (entropy = %.3f)" % self.env.entropy)
                self.fig = fig
                self.ax = ax
            else:
                fig = self.fig
                ax = self.ax
                ax[0].title.set_text("observation map (current: %s)" % self._obs_to_str())
                ax[1].title.set_text("source probability distribution (entropy = %.3f)" % self.env.entropy)
        else:
            fig, ax = self._setup_render()
            ax[0].set_title("observation map (current: %s)" % self._obs_to_str())
            ax[1].set_title("source probability distribution (entropy = %.3f)" % self.env.entropy)

        self._update_render(fig, ax, toptext=toptext)

        if self.video_live:
            plt.pause(0.1)
        plt.draw()
        framefilename = self._framefilename(num)
        fig.savefig(framefilename, dpi=150)
        if not self.video_live:
            plt.close(fig)

    # ________internal___________________________________________________________________
    def _obs_to_str(self, ):
        if self.env.obs["done"]:
            out = "source found"
        else:
            if self.env.draw_source:
                out = "source not found, hit = " + str(self.env.obs["hit"])
            else:
                out = "hit = " + str(self.env.obs["hit"])
        return out

    def _setup_render(self, ):

        figsize = (12.5, 5.5)

        if self.env.Ndim == 1:
            # setup figure
            fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.3, 3]}, figsize=figsize)
            bottom = 0.12
            top = 0.86
            left = 0.08
            right = 0.96
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.36)

            # state
            ax[0].set_facecolor('k')
            ax[0].set_ylim([0, 1])
            ax[0].set_xlim((0 - 0.5, self.env.N - 1 + 0.5))
            ax[0].get_xaxis().set_visible(False)
            if self.env.Nhits <= 6:
                ax[0].set_yticks(np.arange(1, self.env.Nhits + 1) / (self.env.Nhits + 1))
                ax[0].set_yticklabels(np.arange(0, self.env.Nhits))
            else:
                ax[0].set_yticks((1 / (self.env.Nhits + 1), self.env.Nhits / (self.env.Nhits + 1)))
                ax[0].set_yticklabels((0, self.env.Nhits-1))

            # p_source
            ax[1].set_facecolor('k')
            ax[1].get_xaxis().set_visible(False)
            ax[1].set_xlim((0 - 0.5, self.env.N - 1 + 0.5))

            # position source
            if self.log_prob:
                yloc = [1e-3, 1e-3]
            else:
                yloc = [0, 0]
            if self.env.draw_source:
                for i in range(2):
                    ax[i].plot(self.env.source[0], yloc[i], color="r", marker="$+$", markersize=8, clip_on=False,
                               zorder=10000)

        elif self.env.Ndim == 2:
            # setup figure
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            bottom = 0.1
            top = 0.88
            left = 0.05
            right = 0.94
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.35)

            # state
            cmap0 = self._cmap0()
            sm0 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=self.env.Nhits - 0.5), cmap=cmap0)
            divider = make_axes_locatable(ax[0])
            cax0 = divider.append_axes("right", size="5%", pad=0.3)
            fig.colorbar(sm0, cax=cax0, ticks=np.arange(0, self.env.Nhits))
            ax[0].set_aspect("equal", adjustable="box")
            ax[0].axis("off")

            # p_source
            cmap1 = self._cmap1()
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            divider = make_axes_locatable(ax[1])
            cax1 = divider.append_axes("right", size="5%", pad=0.3)
            if self.log_prob:
                cbar1 = fig.colorbar(sm1, cax=cax1, extend="min")
            else:
                cbar1 = fig.colorbar(sm1, cax=cax1)
            if self.video_live:
                self.cbar1 = cbar1
            ax[1].set_aspect("equal", adjustable="box")
            ax[1].axis("off")

            # position of source
            if self.env.draw_source:
                for i in range(2):
                    ax[i].plot(self.env.source[0], self.env.source[1], color="r", marker="$+$", markersize=8, zorder=10000)

        elif self.env.Ndim == 3:
            # setup figure
            fig, ax = plt.subplots(1, 2, figsize=figsize)
            bottom = 0.05
            top = 0.92
            left = 0.04
            right = 0.97
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, hspace=0.0)
            ax[0].remove()
            ax[0] = fig.add_subplot(121, projection='3d')
            ax[1].remove()
            ax[1] = fig.add_subplot(122, projection='3d')

            # draw bounding box
            x0 = -0.5
            x1 = self.env.N - 0.5
            for i in range(2):
                if i == 0:
                    color_cube = "darkgrey"
                    alpha_cube = 0.25
                elif i == 1:
                    color_cube = "darkgrey"
                    alpha_cube = 0.8
                ax[i].set_xlim3d([x0, x1])
                ax[i].set_ylim3d([x0, x1])
                ax[i].set_zlim3d([x0 + self.env.N/10, x1 - self.env.N/10])  # does not seem possible to enforce aspect_ratio=1 with mplot3d, so manual tweaking of zlim
                # back cube edges (split to avoid drawing the same edge twice)
                zorder = 10
                ax[i].plot(
                    [x0, x0, x1],
                    [x0, x1, x1],
                    [x0, x0, x0],
                    color=color_cube,
                    alpha=alpha_cube,
                    zorder=zorder,
                )
                ax[i].plot(
                    [x0, x0, x1],
                    [x0, x1, x1],
                    [x1, x1, x1],
                    color=color_cube,
                    alpha=alpha_cube,
                    zorder=zorder,
                )
                ax[i].plot(
                    [x0, x0],
                    [x1, x1],
                    [x0, x1],
                    color=color_cube,
                    alpha=alpha_cube,
                    zorder=zorder,
                )
                # front cube edges
                zorder = 1e6
                ax[i].plot(
                    [x1, x1, x1, x1, x1, x0, x0, x1],
                    [x0, x1, x1, x0, x0, x0, x0, x0],
                    [x0, x0, x1, x1, x0, x0, x1, x1],
                    color=color_cube,
                    alpha=alpha_cube,
                    zorder=zorder,
                )
                if False:  # for debug
                    ax[i].set_xticks(ticks=np.arange(x0, x1 + 1))
                    ax[i].set_yticks(ticks=np.arange(x0, x1 + 1))
                    ax[i].set_zticks(ticks=np.arange(x0, x1 + 1))
                    ax[i].set_xlabel("x")
                    ax[i].set_ylabel("y")
                    ax[i].set_zlabel("z")
                else:
                    ax[i].axis("off")
                    ax[i].grid(False)

            # state
            cmap0 = self._cmap0()
            alpha0 = self._alpha0()
            sm0 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-0.5, vmax=self.env.Nhits - 0.5), cmap=cmap0)
            cbar0 = fig.colorbar(sm0, ax=ax[0], shrink=0.8, aspect=20, ticks=np.arange(0, self.env.Nhits))
            cbar0.solids.set(alpha=alpha0)

            # source
            cmap1 = self._cmap1()
            alpha1 = self._alpha1()
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            if self.log_prob:
                cbar1 = fig.colorbar(sm1, ax=ax[1], shrink=0.8, aspect=20, extend="min")
            else:
                cbar1 = fig.colorbar(sm1, ax=ax[1], shrink=0.8, aspect=20)
            cbar1.solids.set(alpha=alpha1)
            if self.video_live:
                self.cbar1 = cbar1

            # position of source
            for i in range(2):
                if self.env.draw_source:
                    if i == 0:
                        color = "k"
                    elif i == 1:
                        color = "k"
                    ax[i].plot((self.env.source[0],), (self.env.source[1],), (self.env.source[2],), color=color,
                               marker="$+$", markersize=8, zorder=10000)

        return fig, ax

    def _update_render(self, fig, ax, toptext=''):

        if self.video_live:
            if hasattr(self, 'artists'):
                for artist in range(len(self.artists)):
                    if self.artists[artist] is not None:
                        if isinstance(self.artists[artist], list):
                            for art in self.artists[artist]:
                                art.remove()
                        else:
                            self.artists[artist].remove()

        if self.env.Ndim == 1:
            self._draw_1D(fig, ax)
        elif self.env.Ndim == 2:
            self._draw_2D(fig, ax)
        elif self.env.Ndim == 3:
            self._draw_3D(fig, ax)

        bottomtext = "$\mathcal{L} = \lambda / \Delta x = $" + str(self.env.lambda_over_dx) \
                     + "$\qquad$ $\mathcal{I} = R \Delta t = $" + str(self.env.R_dt) \
                     + "$\qquad$ $h_{\mathrm{init}}$ = " + str(self.env.initial_hit)
        sup = plt.figtext(0.5, 0.99, toptext, fontsize=13, ha="center", va="top")
        bot = plt.figtext(0.5, 0.01, bottomtext, fontsize=10, ha="center", va="bottom")
        if self.video_live:
            self.artists.append(sup)
            self.artists.append(bot)

    def _draw_1D(self, fig, ax):
        # hit map
        cmap0 = self._cmap0()
        mask = self.env.hit_map > - 1
        height = (self.env.hit_map[mask] + 1) / (self.env.Nhits + 1)
        color = self.env.hit_map[mask] / (self.env.Nhits - 1)
        plt0 = ax[0].bar(x=np.arange(self.env.N)[mask], height=height, bottom=0, width=1, color=cmap0(color), clip_on=False, zorder=100)

        # p_source
        if self.log_prob:
            plt1 = ax[1].bar(x=np.arange(self.env.N), height=self.env.p_source, bottom=1e-3, width=1, color="lightgrey",
                             clip_on=False, zorder=100)
            ax[1].set_yscale('log')
            ax[1].set_ylim(bottom=1e-3, top=1)
        else:
            plt1 = ax[1].bar(x=np.arange(self.env.N), height=self.env.p_source, bottom=0, width=1, color="lightgrey",
                             clip_on=False, zorder=100)
            ytop = 1.05 * np.max(self.env.p_source)
            if ytop == 0:
                ytop = 1
            ax[1].set_ylim(bottom=0, top=ytop)

        # position of agent
        if self.log_prob:
            yloc = [1e-3, 1e-3]
        else:
            yloc = [0, 0]
        aloc = [0] * 2
        for i in range(2):
            aloc[i] = ax[i].plot(self.env.agent[0], yloc[i], "ro", clip_on=False, zorder=10000)

        if self.video_live:
            self.artists = [plt0, plt1] + [a for a in aloc]

    def _draw_2D(self, fig, ax):
        # hit map
        cmap0 = self._cmap0()
        img0 = ax[0].imshow(
            np.transpose(self.env.hit_map),
            vmin=-0.5,
            vmax=self.env.Nhits - 0.5,
            origin="lower",
            cmap=cmap0,
        )

        # p_source
        cmap1 = self._cmap1()
        img1 = ax[1].imshow(
            np.transpose(self.env.p_source),
            vmin=np.min(self.env.p_source),
            vmax=np.max(self.env.p_source),
            origin="lower",
            aspect='equal',
            cmap=cmap1,
        )
        if self.video_live:
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            self.cbar1.update_normal(sm1)

        # position of agent
        aloc = [0] * 2
        for i in range(2):
            aloc[i] = ax[i].plot(self.env.agent[0], self.env.agent[1], "ro")

        if self.video_live:
            self.artists = [img0, img1] + [a for a in aloc]

    def _draw_3D(self, fig, ax):
        x0 = -0.5
        x1 = self.env.N - 0.5

        # record trajectory
        if hasattr(self, 'traj'):
            self.traj = np.vstack((self.traj, np.array(self.env.agent)))
        else:
            self.traj = np.array(self.env.agent)

        # *** hit map
        # colored markers for hits
        x, y, z = np.nonzero(self.env.hit_map != -1)
        c = self.env.hit_map[self.env.hit_map != -1]
        c[c == -2] = self.env.Nhits
        alpha0 = self._alpha0()
        cmap0 = self._cmap0()
        s = [15000 / self.env.N ** 2] * len(c)
        bb = np.vstack((x == self.env.agent[0], y == self.env.agent[1], z == self.env.agent[2]))
        test = np.all(bb, axis=0)
        if np.sum(test) > 0:
            index = np.argwhere(test)[0][0]
            s[index] = 3 * s[index]  # marker size is bigger at the agent location
        sc0 = ax[0].scatter(
            x,
            y,
            z,
            cmap=cmap0,
            c=c,
            vmin=-0.5,
            vmax=self.env.Nhits - 0.5,
            depthshade=False,
            s=s,
            alpha=alpha0,
            edgecolors='face',
        )

        # trajectory
        if len(self.traj.shape) == 2:
            plt0 = ax[0].plot(
                self.traj[:, 0],
                self.traj[:, 1],
                self.traj[:, 2],
                color="black",
                alpha=0.5,
                linewidth=0.5,
                zorder=10000,
            )
        else:
            plt0 = None

        # ** p_source
        # draw the matrix slices at the agent location in a way that is equivalent to imshow in 2D
        cmap1 = self._cmap1()
        alpha1 = self._alpha1()
        surf1 = [0] * 3
        for axis in range(3):
            data = np.zeros([self.env.N + 1] * 2)
            if axis == 0:
                if self.marginal_prob_3d:
                    data[0:self.env.N, 0:self.env.N] = np.transpose(np.sum(self.env.p_source, axis=0))
                else:
                    data[0:self.env.N, 0:self.env.N] = np.transpose(self.env.p_source[self.env.agent[0], :, :])
                Y, Z = np.meshgrid(np.arange(-0.5, self.env.N + 0.5), np.arange(-0.5, self.env.N + 0.5))
                X = x0 * np.ones(Y.shape)
            elif axis == 1:
                if self.marginal_prob_3d:
                    data[0:self.env.N, 0:self.env.N] = np.transpose(np.sum(self.env.p_source, axis=1))
                else:
                    data[0:self.env.N, 0:self.env.N] = np.transpose(self.env.p_source[:, self.env.agent[1], :])
                X, Z = np.meshgrid(np.arange(-0.5, self.env.N + 0.5), np.arange(-0.5, self.env.N + 0.5))
                Y = x1 * np.ones(X.shape)
            elif axis == 2:
                if self.marginal_prob_3d:
                    data[0:self.env.N, 0:self.env.N] = np.transpose(np.sum(self.env.p_source, axis=2))
                else:
                    data[0:self.env.N, 0:self.env.N] = np.transpose(self.env.p_source[:, :, self.env.agent[2]])
                X, Y = np.meshgrid(np.arange(-0.5, self.env.N + 0.5), np.arange(-0.5, self.env.N + 0.5))
                Z = x0 * np.ones(X.shape)

            data = data / np.max(self.env.p_source)

            surf1[axis] = ax[1].plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmap1(data), shade=False,
                                             alpha=alpha1)

        if self.video_live:
            if self.log_prob:
                sm1 = plt.cm.ScalarMappable(norm=colors.LogNorm(vmin=1e-3, vmax=1.0), cmap=cmap1)
            else:
                sm1 = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=np.min(self.env.p_source), vmax=np.max(self.env.p_source)), cmap=cmap1)
            self.cbar1.update_normal(sm1)
            self.cbar1.solids.set(alpha=alpha1)

        # trajectory
        if len(self.traj.shape) == 2:
            plt1 = ax[1].plot(
                self.traj[:, 0],
                self.traj[:, 1],
                self.traj[:, 2],
                color="black",
                alpha=1,
                zorder=10000,
            )
        else:
            plt1 = None

        aloc3d = ax[1].plot((self.env.agent[0],), (self.env.agent[1],), (self.env.agent[2],), "o", color="k", zorder=10000)
        aloc2d = [0] * 3
        for axis in range(3):
            if axis == 0:
                aloc2d[0] = ax[1].scatter(x0, self.env.agent[1], self.env.agent[2], c="k", marker="o")
            elif axis == 1:
                aloc2d[1] = ax[1].scatter(self.env.agent[0], x1, self.env.agent[2], c="k", marker="o")
            elif axis == 2:
                aloc2d[2] = ax[1].scatter(self.env.agent[0], self.env.agent[1], x0, c="k", marker="o")

        if self.video_live:
            self.artists = [sc0, plt0] + [surf for surf in surf1] + [plt1, aloc3d] + [aloc for aloc in aloc2d]

    def _cmap0(self):
        topcolors = plt.cm.get_cmap('Greys', 128)
        if self.env.Ndim == 3:
            bottomcolors = plt.cm.get_cmap('jet', 128)
        else:
            bottomcolors = plt.cm.get_cmap('Spectral_r', 128)
        newcolors = np.vstack((topcolors(0.5),
                               bottomcolors(np.linspace(0, 1, self.env.Nhits - 1))))
        cmap0 = ListedColormap(newcolors, name='GreyColors')
        if self.env.Ndim == 2:
            cmap0.set_under(color="black")
        return cmap0

    def _cmap1(self):
        if self.env.Ndim == 1:
            cmap1 = plt.cm.get_cmap("jet", 50)
        elif self.env.Ndim == 2:
            cmap1 = plt.cm.get_cmap("viridis", 50)
        elif self.env.Ndim == 3:
            cmap1 = plt.cm.get_cmap("Blues", 50)
        return cmap1

    def _alpha0(self):
        alpha0 = None
        if self.env.Ndim == 3:
            alpha0 = 0.7
        return alpha0

    def _alpha1(self):
        alpha1 = None
        if self.env.Ndim == 3:
            alpha1 = 0.7
        return alpha1

    def _framefilename(self, num):
        framefilename = os.path.join(self.frame_path,
                                     str(os.path.basename(self.video_path) + "_" + str(num).zfill(8) + ".png"))
        return framefilename

    def _make_video(self, frame_rate, keep_frames):
        out = self.video_path + ".mp4"
        cmd = "ffmpeg -loglevel quiet -r " + str(frame_rate) + " -pattern_type glob -i '" + \
              os.path.join(self.frame_path, "*.png'") + " -c:v libx264 " + out
        exit_code = os.system(cmd)
        if exit_code != 0:
            print("Warning: could not make a video, is ffmpeg installed?")
        else:
            if not keep_frames:
                shutil.rmtree(self.frame_path)
        return exit_code
