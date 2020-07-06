from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable


# +
def get_2d_hist(xyz_nm, size=None, pixel_size=10, z_range=None):
    
    xyz_pos = np.array(xyz_nm)
    x_pos = xyz_pos[:,0]
    y_pos = xyz_pos[:,1]
    z_pos = xyz_pos[:,2]
    
    if z_range is None: 
        z_range = [z_pos.min(),z_pos.max()]
        
    z_pos = np.clip(z_pos,z_range[0],z_range[1])
    z_weight = ((z_pos-z_pos.min())/(z_pos.max()-z_pos.min()))

    if size is None:
        hist_dim = int(x_pos.max()//pixel_size), int(y_pos.max()//pixel_size)
    else:
        hist_dim = int(size[0]//pixel_size), int(size[1]//pixel_size)

    hist = np.histogram2d(x_pos, y_pos, bins=hist_dim, range=[[0, hist_dim[0]*pixel_size], [0, hist_dim[1]*pixel_size]])[0]
    z_hist = np.histogram2d(x_pos, y_pos, bins=hist_dim, range=[[0, hist_dim[0]*pixel_size], [0, hist_dim[1]*pixel_size]], weights=z_weight)[0]
    return hist, z_hist


class RenderHist2D():
    
    def __init__(self, size, pixel_size, sigma_blur, clip_percentile):
        
        self.size = size
        self.pixel_size = pixel_size
        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile
        
    def plot(self, xyz_nm, figsize=(10,10)):
            
        hist, _ = get_2d_hist(xyz_nm, self.size, self.pixel_size)
        hist = np.clip(hist,0,np.percentile(hist,self.clip_percentile))

        if self.sigma_blur:
            hist = gaussian_filter(hist, sigma=[self.sigma_blur/self.pixel_size,self.sigma_blur/self.pixel_size])

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(hist)

class RenderHist3D():
    
    def __init__(self, size, z_range, pixel_size, sigma_blur, clip_percentile, gamma):
        
        self.size = size
        self.z_range = z_range
        self.pixel_size = pixel_size
        self.sigma_blur = sigma_blur
        self.clip_percentile = clip_percentile
        self.gamma = gamma
        
    def plot(self, xyz_nm, figsize=(10,10), fontsize=(15)):
        
        hist, z_hist = get_2d_hist(xyz_nm, self.size, self.pixel_size, self.z_range)
        with np.errstate(divide='ignore', invalid='ignore'):
            z_avg = z_hist/hist
        
        hist = np.clip(hist,0,np.percentile(hist,self.clip_percentile))
        z_avg[np.isnan(z_avg)] = 0

        val = (hist-hist.min())/(hist.max()-hist.min())
        sat = np.ones(hist.shape)
        hue = z_avg
        
        HSV = np.concatenate((hue[:,:,None],sat[:,:,None],val[:,:,None]),-1)
        RGB = hsv_to_rgb(HSV)**(1/self.gamma)       
        
        if self.sigma_blur:
            RGB = np.array([gaussian_filter(RGB[:,:,i], sigma=[self.sigma_blur/self.pixel_size,self.sigma_blur/self.pixel_size]) for i in range(3)]).transpose(1,2,0)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        im = ax.imshow(RGB, cmap='hsv')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.25, pad=-0.25)
        colb = plt.colorbar(im, cax=cax, orientation='vertical', ticks=[])   
        colb.outline.set_visible(False)
        
        cax.text(0.12,0.04,f'{self.z_range[0]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
        cax.text(0.12,0.88,f'{self.z_range[1]} nm', rotation=90, color='white', fontsize=15, transform=cax.transAxes)
