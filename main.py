import numpy as np
import matplotlib.pyplot as plt
import os
import netCDF4 as nc

import data_flie_transform as dft
import npdata_visualize as nvl
import argparse
from glob import glob
    
# multi process nc file -> npdata
class MultiXNcFileToNpdata_Pop1(dft.FileToData):
    def __call__(self, file_folder, file_prefix_name, fea_name):
        import netCDF4 as nc
        import glob

        file_folder_name_list = glob.glob(os.path.join(file_folder, f"{file_prefix_name}*"))
        file_folder_name_list = sorted(file_folder_name_list)
        npdata_list = []
        for index, file_folder_name in enumerate(file_folder_name_list):
            with nc.Dataset(file_folder_name, "r") as ncdata:
                if index != len(file_folder_name_list)-1:
                    npdata = ncdata.variables[fea_name][:, :, :,  :-1]
                else:
                    npdata = ncdata.variables[fea_name][:, :, :,  :]
                npdata_list.append(npdata)
        npdata = np.concatenate(npdata_list, axis=-1)

        return npdata
    
# multi process nc file -> npdata
class NpdataDim0Interp(dft.DataToData):
    def __call__(self, npdata):
        length = len(npdata)
        interp_npdata = np.zeros(shape=(length-1, *npdata.shape[1:]))
        for i in range(length-1):
            interp_npdata[i] = (npdata[i] + npdata[i+1]) / 2

        return interp_npdata

# p, b, s npdata -> rho npdata
class SNpdataToSOD(dft.DataToData):
    # SOD: sea obstale distribution
    def __call__(self, npdata_s):
        npdata_sod = np.zeros_like(npdata_s)
        npdata_sod[npdata_s!=0] = 1.0
        return npdata_sod

# multi process nc file -> npdata
class YuanNpdataToNcFile(dft.DataToData):
    def __call__(self, file_folder, file_name, npdata_dict, bathy_file_id):
        # 创建一个空的 NetCDF 文件对象
        ncfile = nc.Dataset(os.path.join(file_folder, file_name), "w", format="NETCDF4")
        len_t, len_z, len_x = npdata_dict["b"].shape

        # 创建维度
        ncfile.createDimension("T", len_t)
        ncfile.createDimension("Z", len_z)
        ncfile.createDimension("X", len_x)

        # 创建变量
        ncdata_t = ncfile.createVariable("T", "f4", ("T",))
        ncdata_z = ncfile.createVariable("Z", "f4", ("Z",))
        ncdata_x = ncfile.createVariable("X", "f4", ("X",))
        ncdata_s = ncfile.createVariable("S", "f4", ("T", "Z", "X"))
        ncdata_b = ncfile.createVariable("Temp", "f4", ("T", "Z", "X"))
        ncdata_u = ncfile.createVariable("U", "f4", ("T", "Z", "X"))
        ncdata_w = ncfile.createVariable("W", "f4", ("T", "Z", "X"))
        ncdata_sod = ncfile.createVariable("sod", "f4", ("Z", "X"))

        # 设置变量数据
        ncdata_t[:] = np.arange(len_t) * npdata_dict["dis_t"]
        ncdata_z[:] = np.arange(len_z) * npdata_dict["dis_z"] + npdata_dict["dis_z"] / 2
        ncdata_x[:] = np.arange(len_x) * npdata_dict["dis_x"] + npdata_dict["dis_x"] / 2

        ncdata_s[:] = npdata_dict["s"]
        ncdata_b[:] = npdata_dict["b"]
        ncdata_u[:] = npdata_dict["u"]
        ncdata_w[:] = npdata_dict["w"]
        ncdata_sod[:] = npdata_dict["sod"]

        # 创建全局属性
        ncatts = {
            "viscC2Leith": npdata_dict["visc_c2_leith"],
            "viscC2LeithD": npdata_dict["visc_c2_leith"],
            "bathyFile": f"topo_norm{bathy_file_id}.binary",
            "OB_Tide_WEST_u_Amp": npdata_dict["tide_amp"],
        }
        ncfile.setncatts(ncatts)

        # 关闭文件
        ncfile.close()

def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--id", type=str)
    parser.add_argument("--cp_id", type=str)
    parser.add_argument("--tide_amp", type=float, default=1e-2)
    parser.add_argument("--visc_c2_leith", type=float, default=2.0)
    parser.add_argument("--dis_t", type=int, default=600)
    parser.add_argument("--dis_z", type=int, default=-4)
    parser.add_argument("--dis_x", type=int, default=200)
    parser.add_argument("--comparing_dis_z", type=int, default=4)
    parser.add_argument("--comparing_dis_x", type=int, default=200)
    parser.add_argument("--t_start", type=int, default=None)
    parser.add_argument("--t_end", type=int, default=None)
    parser.add_argument("--z_start", type=int, default=0)
    parser.add_argument("--z_end", type=int, default=128)
    parser.add_argument("--x_start", type=int, default=2000-512)
    parser.add_argument("--x_end", type=int, default=2000+512)
    parser.add_argument("--bathy_file_id", type=int, default=1)

    parser.set_defaults(id=52)
    parser.set_defaults(cp_id=43)
    parser.set_defaults(tide_amp=0.2)
    parser.set_defaults(bathy_file_id=13)
    parser.set_defaults(visc_c2_leith=5)
    parser.set_defaults(cp_id=42)
    
    args = parser.parse_args()
    return args

def main():

    args = get_args()

    exp_folders = glob(f"exp{args.id}*")
    for i in range(len(exp_folders)):
        if exp_folders[i][-3:] != "zip":
            exp_folder = exp_folders[i]
        if i > 0:
            raise f"please make sure only one exp {args.id}"
    print(f"exp_folder: {exp_folder}")

    comparing_exp_folders = glob(f"exp{args.cp_id}*")
    for i in range(len(comparing_exp_folders)):
        if comparing_exp_folders[i][-3:] != "zip":
            comparing_exp_folder = comparing_exp_folders[i]
        if i > 0:
            raise f"please make sure only one exp {args.cp_id}"
    print(f"comparing_exp_folder: {comparing_exp_folder}")

    program_folder = exp_folder
    comparing_program_folder = comparing_exp_folder
    file_folder = os.path.join(program_folder, "run")
    dim_slice = (slice(None, None), slice(args.z_start, args.z_end), 0, slice(args.x_start, args.x_end))
    frame_folder = os.path.join(program_folder, "frame")
    movie_folder = os.path.join(program_folder, "movie")
    npdata_fea_names = ["b", "u", "w", "s"]

    tide_amp = args.tide_amp

    dis_t = args.dis_t
    dis_z = args.dis_z
    dis_x = args.dis_x
    visc_c2_leith = args.visc_c2_leith
    bathy_file_id = args.bathy_file_id

    def read_nc_save_npz():
        transform = MultiXNcFileToNpdata_Pop1()
        npdata_u = transform(file_folder=file_folder, file_prefix_name="stateU", fea_name="U")
        transform = dft.MultiXNcFileToNpdata()
        npdata_b = transform(file_folder=file_folder, file_prefix_name="stateTemp", fea_name="Temp")
        npdata_s = transform(file_folder=file_folder, file_prefix_name="stateS", fea_name="S")
        npdata_w = transform(file_folder=file_folder, file_prefix_name="stateW", fea_name="W")

        transform = dft.NpdataDictToNpzFile()
        npdata_dict = {"u": npdata_u, "b": npdata_b, "s": npdata_s, "w": npdata_w}
        transform(npdata_dict=npdata_dict, file_folder=program_folder, file_name="data.npz")

        transform = dft.NpdataToMatFile()
        transform(npdata=npdata_b, file_folder=program_folder, file_name="b.mat", fea_name="b")


    def interp_cliped_data_save_npz():
        transform = dft.NpzFileToNpdataDict()
        npdata_dict = transform(file_folder=program_folder, file_name="data.npz")

        transform = NpdataDim0Interp()

        npdata_u = npdata_dict["u"]

        npdata_u = npdata_u.transpose(3, 1, 2, 0)
        npdata_u = transform(npdata_u)
        npdata_u = npdata_u.transpose(3, 1, 2, 0)

        npdata_w = npdata_dict["w"]
        npdata_w = npdata_w.transpose(1, 0, 2, 3)
        npdata_w = transform(npdata_w)
        npdata_w = npdata_w.transpose(1, 0, 2, 3)

        npdata_s = npdata_dict["s"]

        npdata_dict["u"] = npdata_u
        npdata_dict["w"] = npdata_w

        for fea_name in npdata_dict:
            npdata_dict[fea_name] = npdata_dict[fea_name][*dim_slice]

        transform = SNpdataToSOD()
        npdata_s = npdata_dict["s"]
        npdata_sod = transform(npdata_s[10])

        npdata_dict["sod"] = npdata_sod
        npdata_s[npdata_s==0.0] = 33.0
        npdata_dict["s"] = npdata_s

        npdata_dict["tide_amp"] = np.array(tide_amp)

        npdata_dict["dis_x"] = np.array(dis_x)
        npdata_dict["dis_z"] = np.array(dis_z)
        npdata_dict["dis_t"] = np.array(dis_t)
        npdata_dict["visc_c2_leith"] = np.array(visc_c2_leith)

        transform = dft.NpdataDictToNpzFile()
        transform(npdata_dict=npdata_dict, file_folder=program_folder, file_name="interp_cliped_data.npz")

        transform = YuanNpdataToNcFile()
        transform(file_folder="ncfile", file_name=f"{program_folder}.nc", npdata_dict=npdata_dict, bathy_file_id=bathy_file_id)

        pass

    def visualize_gen_movie():
        transform = dft.NpzFileToNpdataDict()
        npdata_dict = transform(file_folder=program_folder, file_name="interp_cliped_data.npz")

        draw_fun = nvl.DrawHeatImage(colorbar_orientation="horizontal", interpolation="spline16", colorbar_aspect=20, aspect=1)
        draw_fun = nvl.DrawContourfImage(colorbar_orientation="horizontal", colorbar_aspect=20)
        tick_info_fun = nvl.CreatePlanarTickInfo()
        tick_info = tick_info_fun(npdata=npdata_dict["b"][0], x_dis=200, y_dis=-4, reduce_alpha=(100, 40), xlabel="x(m)", ylabel="z(m)")
        image_generator = nvl.SingleImageGenerator(figsize=(30, 12), fontsize=24, draw_fun=draw_fun, tick_info=tick_info)
        image_generator(npdata_list=npdata_dict["sod"], frame_folder=frame_folder, frame_name="sod.jpg")
        movie_generator = nvl.MovieGenerator(image_generator=image_generator, frame_rate=25)


        # for fea_name in npdata_dict:
        #     movie_generator(npdata_list=npdata_dict[fea_name], movie_folder=movie_folder, movie_name=f"{fea_name}.mp4")
        movie_generator(npdata_list=npdata_dict["b"], movie_folder=movie_folder, movie_name=f"b.mp4")
        movie_generator(npdata_list=npdata_dict["u"], movie_folder=movie_folder, movie_name=f"u.mp4")
        movie_generator(npdata_list=npdata_dict["s"], movie_folder=movie_folder, movie_name=f"s.mp4")
        movie_generator(npdata_list=npdata_dict["w"], movie_folder=movie_folder, movie_name=f"w.mp4")

    def comparing_visualize_gen_movie():
        movie_folder = os.path.join(program_folder, f"movie_comparing_{comparing_program_folder}")
        transform = dft.NpzFileToNpdataDict()
    
        npdata_dict = transform(file_folder=program_folder, file_name="interp_cliped_data.npz")
        comparing_npdata_dict = transform(file_folder=comparing_program_folder, file_name="interp_cliped_data.npz")

        draw_fun = nvl.DrawHeatImage(aspect=2)
        tick_info_fun = nvl.CreatePlanarTickInfo()
        tick_info = tick_info_fun(npdata=npdata_dict["b"][0], x_dis=200, y_dis=-4, reduce_alpha=(100, 40), xlabel="x(m)", ylabel="z(m)")
        comparing_tick_info = tick_info_fun(npdata=comparing_npdata_dict["b"][0], x_dis=200, y_dis=-4, reduce_alpha=(100, 40), xlabel=None, ylabel="z(m)")

        draw_funs = [draw_fun, draw_fun]
        tick_infos = [comparing_tick_info, tick_info]

        image_generator = nvl.ComeparingColorbarImageGenerator(figsize=(30, 24), fontsize=30, draw_funs=draw_funs, tick_infos=tick_infos, 
                                                                colorbar_orientation="horizontal", colorbar_aspect=20)
        
        movie_generator = nvl.MovieGenerator(image_generator=image_generator, frame_rate=10)

        for fea in npdata_fea_names:
            npdata_list = [[comparing_npdata_dict[fea][t_id], npdata_dict[fea][t_id]] for t_id in range(npdata_dict[fea].shape[0])]
            movie_generator(npdata_list=npdata_list, movie_folder=movie_folder, movie_name=f"{fea}.mp4")

    read_nc_save_npz()
    interp_cliped_data_save_npz()
    # visualize_gen_movie()
    comparing_visualize_gen_movie()

    pass

def test():
    import netCDF4 as nc
    with nc.Dataset(os.path.join("ncfile", "exp08_30_data_amp02_topo1_visc_c2_leith_2.nc"), "r") as ncdata:
        npdata_t = ncdata.variables["T"][:]
        npdata_z = ncdata.variables["Z"][:]
        npdata_x = ncdata.variables["X"][:]
        npdata_s = ncdata.variables["S"][:]
        npdata_b = ncdata.variables["Temp"][:]
        npdata_u = ncdata.variables["U"][:]
        npdata_w = ncdata.variables["W"][:]

    pass

if __name__ == "__main__":
    main()

