import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod

import data_flie_transform as dft

class FramesToMovie:

    def __call__(self, frame_folder, movie_folder, movie_name, frame_rate):
        frame_pattern = os.path.join(frame_folder, "*.png")
        os.system(f"rm -rf {os.path.join(movie_folder, movie_name)}")
        os.system(f"ffmpeg -framerate {frame_rate} -pattern_type glob"
                  f" -i \"{frame_pattern}\" -c:v libx264 -r 30 -pix_fmt yuv420p"
                  f" {os.path.join(movie_folder, movie_name)}")
        print(f"saving movie to {os.path.join(movie_folder, movie_name)}")

class TickInfo(ABC):
    def __init__(self, x_ticks=None, y_ticks=None, x_ticks_labels=None, y_ticks_labels=None, xlabel=None, ylabel=None):
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.x_ticks_labels = x_ticks_labels
        self.y_ticks_labels = y_ticks_labels
        self.xlabel = xlabel
        self.ylabel = ylabel

    def apply(self, ax, fontsize):
        pass

class PlanarTickInfo(TickInfo):
    def __init__(self, x_ticks=None, y_ticks=None, x_ticks_labels=None, y_ticks_labels=None, xlabel=None, ylabel=None):
        self.x_ticks = x_ticks
        self.y_ticks = y_ticks
        self.x_ticks_labels = x_ticks_labels
        self.y_ticks_labels = y_ticks_labels
        self.xlabel = xlabel
        self.ylabel = ylabel

    def apply(self, ax, fontsize):
        if self.x_ticks.any() != None:
            ax.set_xticks(self.x_ticks)
            ax.set_xticklabels(self.x_ticks_labels, fontsize=fontsize)
        if self.y_ticks.any() != None:
            ax.set_yticks(self.y_ticks)
            ax.set_yticklabels(self.y_ticks_labels, fontsize=fontsize)
        if self.xlabel:
            ax.set_xlabel(self.xlabel, fontsize=fontsize)
        if self.ylabel:
            ax.set_ylabel(self.ylabel, fontsize=fontsize)

class CreateTickInfo(ABC):
    def __call__(self):
        pass


class CreatePlanarTickInfo(CreateTickInfo):
    def __call__(self, npdata, x_dis, y_dis, reduce_alpha, xlabel, ylabel):
        if isinstance(reduce_alpha, tuple):
            reduce_alpha_x, reduce_alpha_y = reduce_alpha
        else:
            reduce_alpha_x = reduce_alpha_y = reduce_alpha
        x_ticks = np.arange(npdata.shape[1])[::reduce_alpha_x]
        x_ticks_labels = x_ticks * x_dis
        x_ticks_labels = [ round(lable, 2) for lable in x_ticks_labels]
        y_ticks = np.arange(npdata.shape[0])[::reduce_alpha_y]
        y_ticks_labels = y_ticks * y_dis
        y_ticks_labels = [ round(lable, 2) for lable in y_ticks_labels]
        return PlanarTickInfo(x_ticks=x_ticks, y_ticks=y_ticks, x_ticks_labels=x_ticks_labels, y_ticks_labels=y_ticks_labels, xlabel=xlabel, ylabel=ylabel)


class DrawFun(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def __call__(self, fig, ax, frame, vmin=None, vmax=None, ticks=None, label=None):
        pass

class DrawHeatImage(DrawFun):
    def __init__(self, cmap="RdBu", interpolation=None, colorbar_orientation=None, colorbar_aspect=20, aspect=1):
        self.cmap = cmap
        self.interpolation = interpolation # spline16
        self.colorbar_orientation = colorbar_orientation # vertical and horizontal
        self.colorbar_aspect = colorbar_aspect
        self.aspect = aspect

    def __call__(self, fig, ax, frame, vmin, vmax):
        im = ax.imshow(frame, vmin=vmin, vmax=vmax, cmap=self.cmap, interpolation=self.interpolation, aspect=self.aspect)

        if self.colorbar_orientation:
            cbar = fig.colorbar(im, orientation=self.colorbar_orientation, aspect=self.colorbar_aspect)

        return im


class DrawHeatContourImage(DrawFun):
    def __init__(self, cmap="RdBu", interpolation=None, colorbar_orientation=None, colorbar_aspect=20, level_num=12, contour_color="black"):
        self.cmap = cmap
        self.interpolation = interpolation # spline16
        self.colorbar_orientation = colorbar_orientation # vertical and horizontal
        self.colorbar_aspect = colorbar_aspect
        self.level_num = level_num
        self.contour_color = contour_color

    def __call__(self, fig, ax, frame, vmin, vmax):
        levels = np.linspace(vmin, vmax, self.level_num) if vmin != vmax else self.level_num
        im = ax.imshow(frame, vmin=vmin, vmax=vmax, cmap=self.cmap, interpolation=self.interpolation)
        ax.contour(frame, levels=levels, colors=self.contour_color)

        if self.colorbar_orientation:
            cbar = fig.colorbar(im, orientation=self.colorbar_orientation, aspect=self.colorbar_aspect)

        return im

class DrawContourfImage(DrawFun):
    def __init__(self, level_num=12, colorbar_orientation=None, colorbar_aspect=20):
        self.level_num = level_num
        self.colorbar_orientation = colorbar_orientation
        self.colorbar_aspect = colorbar_aspect
    def __call__(self, fig, ax, frame, vmin, vmax):
        levels = np.linspace(vmin, vmax, self.level_num) if vmin != vmax else self.level_num
        im = ax.contourf(frame, levels=levels)
        ax.invert_yaxis()
        if self.colorbar_orientation:
            cbar = fig.colorbar(im, orientation=self.colorbar_orientation, aspect=self.colorbar_aspect)

        return im

class DrawVectorImage(DrawFun):
    def __init__(self):
        pass
    def __call__(self, fig, ax, frame, vmin, vmax):
        pass

class DrawStreamPlotImage(DrawFun):
    def __init__(self, linewidth=2, density=3):
        self.linewidth = linewidth
        self.density = density
        
    def __call__(self, fig, ax, frame, vmin, vmax):
        x = range(frame.shape[1])
        y = range(frame.shape[0])
        x, y = np.meshgrid(x, y)
        y, x = np.mgrid[0:len(frame):len(frame)*1j, 0:len(frame[0]):len(frame[0])*1j]

        u, v = frame[:, :, 0], frame[:, :, 1]
        ax.streamplot(x=x, y=y, u=u, v=v, linewidth=self.linewidth, cmap=plt.cm.viridis, density=self.density)
        ax.invert_yaxis()

class DrawCurveImage(DrawFun):
    def __init__(self):
        pass
    def __call__(self, fig, ax, frame, vmin, vmax):
        pass

class ImageGenerator(ABC):
    @abstractmethod
    def __call__(self):
        pass

class SingleImageGenerator(ImageGenerator):
    def __init__(self, figsize, fontsize, draw_fun, tick_info=TickInfo()):
        self.figsize = figsize
        self.tick_info = tick_info
        self.draw_fun = draw_fun
        self.fontsize = fontsize

    def set_figsize(self, figsize):
        self.figsize = figsize

    def set_draw_fun(self, draw_fun):
        self.draw_fun = draw_fun

    def __call__(self, npdata_list, frame_folder, frame_name, title=None, vmin=None, vmax=None):
        npdata = npdata_list
        plt.rcParams.update({'font.size': self.fontsize})
        os.makedirs(frame_folder, exist_ok=True)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        vmin = np.min(npdata) if vmin == None else vmin
        vmax = np.max(npdata) if vmax == None else vmax
        self.draw_fun(fig, ax, npdata, vmin, vmax)
        plt.title(title, fontsize=self.fontsize)
        self.tick_info.apply(ax, self.fontsize)
        fig.savefig(os.path.join(frame_folder, frame_name))
        plt.cla()
        matplotlib.pyplot.close()

class ComeparingColorbarImageGenerator(ImageGenerator):
    def __init__(self, figsize, fontsize, draw_funs=[], tick_infos=[], colorbar_orientation=None, colorbar_aspect=10):
        self.figsize = figsize
        self.tick_infos = tick_infos
        self.draw_funs = draw_funs
        self.colorbar_orientation = colorbar_orientation
        self.colorbar_aspect = colorbar_aspect
        self.__clear_colorbar()
        self.fontsize = fontsize

    def __clear_colorbar(self):
        for draw_fun in self.draw_funs:
            draw_fun.colorbar_orientation=None

    def set_draw_funs(self, draw_funs):
        self.draw_funs = draw_funs
        self.__clear_colorbar()

    def __call__(self, npdata_list, frame_folder, frame_name, vmin=None, vmax=None, title=None):
        plt.rcParams.update({'font.size': self.fontsize})
        os.makedirs(frame_folder, exist_ok=True)
        image_num = len(npdata_list)
        fig, axs = plt.subplots(nrows=image_num, ncols=1, figsize=self.figsize)
        vmin = min([np.min(npdata) for npdata in npdata_list]) if vmin == None else vmin
        vmax = max([np.max(npdata) for npdata in npdata_list]) if vmax == None else vmax
        ims = []
        for index in range(image_num):
            im = self.draw_funs[index](fig, axs[index], npdata_list[index], vmin, vmax)
            ims.append(im)
            self.tick_infos[index].apply(axs[index], self.fontsize)
        axs[0].set_title(title, fontsize=self.fontsize)

        fig.colorbar(ims[-1], ax=axs, orientation=self.colorbar_orientation, aspect = self.colorbar_aspect)

        fig.savefig(os.path.join(frame_folder, frame_name))
        plt.cla()
        matplotlib.pyplot.close()

class ComeparingColorbar2ImageGenerator(ImageGenerator):
    def __init__(self, figsize, fontsize, draw_funs=[], tick_infos=[]):
        self.figsize = figsize
        self.tick_infos = tick_infos
        self.draw_funs = draw_funs
        self.colorbar_orientation = "horizontal"
        self.colorbar_aspect = 20
        self.__clear_colorbar()
        self.fontsize = fontsize

    def __clear_colorbar(self):
        for draw_fun in self.draw_funs:
            draw_fun.colorbar_orientation=None

    def set_draw_funs(self, draw_funs):
        self.draw_funs = draw_funs
        self.__clear_colorbar()

    def __call__(self, npdata_list, frame_folder, frame_name, vmin=None, vmax=None, title=None):
        plt.rcParams.update({'font.size': self.fontsize})
        os.makedirs(frame_folder, exist_ok=True)
        image_num = len(npdata_list)
        fig, axs = plt.subplots(nrows=image_num, ncols=1, figsize=self.figsize)
        vmin = min([np.min(npdata) for npdata in npdata_list]) if vmin == None else vmin
        vmax = max([np.max(npdata) for npdata in npdata_list]) if vmax == None else vmax
        ims = []
        for index in range(image_num):
            im = self.draw_funs[index](fig, axs[index], npdata_list[index], vmin, vmax)
            ims.append(im)
            # self.tick_infos[index].apply(axs[index], self.fontsize)
        axs[0].set_title(title, fontsize=self.fontsize)

        cbaxes = fig.add_axes([0.1, 0, .82, 0.05])
        fig.colorbar(ims[-1], ax=axs, orientation=self.colorbar_orientation, aspect = self.colorbar_aspect, pad=0, cax=cbaxes)

        fig.savefig(os.path.join(frame_folder, frame_name))
        plt.cla()
        matplotlib.pyplot.close()

class MovieGenerator:
    def __init__(self, image_generator, frame_rate):
        self.image_generator = image_generator
        self.frame_rate = frame_rate
        self.frames_to_movie = FramesToMovie()

    def __call__(self, npdata_list, movie_folder, movie_name, vmin=None, vmax=None):
        fea_name = ".".join(movie_name.split(".")[:-1])
        frame_folder = os.path.join(movie_folder, f"frame_{fea_name}")
        os.makedirs(frame_folder, exist_ok=True)

        for t_id in tqdm(range(len(npdata_list))):
            frame_name = f"frame_{fea_name}_{str(t_id).rjust(len(str(len(npdata_list))), '0')}.png"
            self.image_generator(npdata_list=npdata_list[t_id], frame_folder=frame_folder, frame_name=frame_name, vmin=vmin, vmax=vmax, title=f"t={t_id}T")

        self.frames_to_movie(frame_folder=frame_folder, movie_folder=movie_folder, movie_name=movie_name, frame_rate=self.frame_rate)

def test():
    data_file_folder = os.path.join("data", "npz_file")
    transform_fun = dft.NpzFileToNpdataDict()
    npdata_dict = transform_fun(file_folder=data_file_folder, file_name="data.npz")
    image_folder = "image"
    movie_folder = "movie"

    dim_slice = (slice(None), slice(None), 0, slice(None))

    npdata_s = npdata_dict["s"][*dim_slice]
    npdata_b = npdata_dict["b"][*dim_slice]

    draw_fun = DrawHeatImage(colorbar_orientation="vertical", colorbar_aspect=10)
    tick_fun = CreatePlanarTickInfo()
    tick_info = tick_fun(npdata_b[0], x_dis=0.2, y_dis=-10, reduce_alpha=32, xlabel="x(km)", ylabel="z(m)")

    image_generator = SingleImageGenerator(figsize=(60, 10), fontsize=30, draw_fun=draw_fun, tick_info=tick_info)
    image_generator(npdata=npdata_b[10], frame_folder=image_folder, frame_name="b_vertical_heat.jpg", title="b(t=10)")
    movie_generator = MovieGenerator(image_generator, 1)
    movie_generator(npdata=npdata_b, movie_folder=movie_folder, movie_name="s.mp4")

    # draw_fun = DrawHeatImage(colorbar_orientation="horizontal", colorbar_aspect=20)
    # image_generator.set_draw_fun(draw_fun)
    # image_generator.set_figsize(figsize=(20, 8))
    # image_generator(npdata=npdata_b[10], frame_path=image_path, frame_name="b_horizontal_heat.jpg", title="b(t=10)")
    # image_generator.set_figsize(figsize=(30, 5))

    # draw_fun = DrawHeatContourImage(colorbar_orientation="vertical", colorbar_aspect=10)
    # image_generator.set_draw_fun(draw_fun)
    # image_generator(npdata=npdata_b[10], frame_path=image_path, frame_name="b_vertical_heat_contour.jpg", title="b(t=10)")

    # draw_fun = DrawContourfImage(colorbar_orientation="vertical", colorbar_aspect=10)
    # image_generator.set_draw_fun(draw_fun)
    # image_generator(npdata=npdata_b[10], frame_path=image_path, frame_name="b_vertical_contourf.jpg", title="b(t=10)")

    # draw_funs = [DrawHeatImage(), DrawHeatImage(interpolation="spline16")]
    # tick_info1 = tick_fun(npdata_b[0], x_dis=0.2, y_dis=-10, reduce_alpha=48, xlabel=None, ylabel="z(m)")
    # tick_info2 = tick_fun(npdata_b[0], x_dis=0.2, y_dis=-10, reduce_alpha=48, xlabel="x(km)", ylabel="z(m)")
    # tick_infos = [tick_info1, tick_info2]
    # image_generator = ComeparingColorbarImageGenerator(figsize=(60, 10), fontsize=18, draw_funs=draw_funs, tick_infos=tick_infos, colorbar_orientation="vertical", colorbar_aspect=10)
    # image_generator(npdata=np.stack([npdata_b[10], npdata_b[15]], axis=0), frame_path=image_path, frame_name="b_comeparing_vertical_heat.jpg", title=["s in 10 and 15"])

def main():
    test()

if __name__ == "__main__":
    main()
