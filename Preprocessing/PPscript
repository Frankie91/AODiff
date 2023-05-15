import numpy as np
import glob
import xarray as xr
from scipy.interpolate import griddata

def blockshaped(arr, nrows, ncols):

    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

# Main Training Data - Popcorn Tiles Creation

border_pixel = 24
tile_size_x = 24*6
tile_size_y = 24*6
x_size = tile_size_x*32
y_size = tile_size_y*27
threshold = round(((tile_size_x*tile_size_y)/100)*10)

popcorn_files = []
# PopcornOriginal/EUSA/**/
# D:\\DANIELE DATA\\PopcornOriginal\\WUSA\\**\\
for i in glob.glob('PopcornOriginal/SAFR/01/*_002.nc', recursive=True):
    popcorn_ds_in = xr.open_dataset(i)
    popcorn_ds = popcorn_ds_in.data_vars['AOD'].values[0,:,:]
    popcorn_ds = popcorn_ds.reshape(1,popcorn_ds.shape[0],popcorn_ds.shape[1])
    mask = popcorn_ds_in.data_vars['bestDataMask'].values
    mask = mask.reshape(1,mask.shape[0],mask.shape[1])
    masked_ds = np.concatenate((popcorn_ds,mask), axis = 0)
    if masked_ds.shape[1] > 3500: # exclude images taken too far north/south
        masked_ds = masked_ds[:,:y_size,
                                border_pixel:x_size+border_pixel]
        masked_ds[0][np.isnan(masked_ds[0])] = 0
        masked_ds[0][np.where(masked_ds[0] > 1.0)] = 1.0
        masked_ds[0][np.where(masked_ds[0] < 0)] = 0
        masked_ds[0][np.where(masked_ds[1] == 0)] = 0
        if np.count_nonzero(masked_ds[0]) > threshold:
        # at least 10% of the pixels nonzero
            masked_ds[0] = masked_ds[0].round(5)
            masked_ds_fin = masked_ds[0].reshape(1,y_size,x_size)
            popcorn_files.append(masked_ds_fin)
            del popcorn_ds, mask, masked_ds,masked_ds_fin

popcorn_base = np.concatenate(popcorn_files, axis = 0)

# np.save('popcorn_base_ASAmin.npy', popcorn_base, allow_pickle=True)

del popcorn_files

# tiles generator

# LandSeaMask = LandSeaMask.reshape(1,666,703)

# masked_dataset = np.concatenate((popcorn_base,LandSeaMask), axis = 0)
# masked_dataset = masked_dataset[:,:630,30:690]
# np.save('popcorn_base.npy', popcorn_base, allow_pickle=True)  

# Due to need for match between AOD and Auxiliary, this part should be executed
# in the DataImport section and not here.

popcorn_tiles= []

for i in range(popcorn_base.shape[0]): 
    popcorn_tiles_generator = blockshaped(popcorn_base[i,:,:],
                                          tile_size_y,tile_size_x)
    popcorn_tiles.append(popcorn_tiles_generator)
    del popcorn_tiles_generator

del popcorn_base  
# remove tiles with too many zeroes

threshold_fin = round((tile_size_x*tile_size_y)-(((tile_size_x*tile_size_y)/100))/2)

suff_array = []

for i in popcorn_tiles:
    for a in range(i.shape[0]):
        i[a,:,:] = i[a,:,:].round(4)
        if np.count_nonzero(i[a,:,:]) > threshold_fin: # 1%
            suff_array.append(i[a,:,:])


# Interpolate missing values

x = np.arange(0, suff_array[0].shape[1])
y = np.arange(0, suff_array[0].shape[0])
xx, yy = np.meshgrid(x, y)

interp_arr = []

for i in suff_array:
    i[i == 0] = np.nan
    i[i < 0.0] = np.nan
    i = np.ma.masked_invalid(i)
    x1 = xx[~i.mask]    
    y1 = yy[~i.mask]
    i = i[~i.mask]
    try: 
        i = griddata((x1, y1), i.ravel(), (xx, yy), method='linear')
    except:
        pass
    else:
        i = i.round(4)
        interp_arr.append(i)    
    
# interp_arr_final = []

# for i2 in interp_arr:
#     i2[i2 <= 0.0] = np.nan
#     i2 = np.ma.masked_invalid(i2)
#     x1 = xx[~i2.mask]
#     y1 = yy[~i2.mask]
#     i2 = i2[~i2.mask]
#     i2 = griddata((x1, y1), i2.ravel(), (xx, yy), method='linear')
#     i2 = i2.round(4)
#     interp_arr_final.append(i2)     

cleanup = []

for i in interp_arr:
    i[i <= 0.0] = np.nan
    i[np.isnan(i)] = 0
    if np.count_nonzero(i == 0) < 1 :
        cleanup.append(i)

np.save('AOD_SAFRmin01.npy', cleanup, allow_pickle=True) 
print(len(cleanup)) 
         
# separate between training and testing data

# AOD_Train, AOD_Test = np.split(interp_arr_final, [int(0.95 * len(interp_arr_final))])

# np.save('C:\\Users\\Dfran\\.spyder-py3\\AOD_Train_Large.npy', AOD_Train, allow_pickle=True)
# np.save('C:\\Users\\Dfran\\.spyder-py3\\AOD_Test_Large.npy', AOD_Test , allow_pickle=True) 
