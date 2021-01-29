import numpy as np
import glob

from PIL import Image

#---------------------

# Hongming's image_convert function to create pngs
def image_convert(name, image_data):
    """
       This function writes a PNG file from a numpy array.
       Args:
       name: Name of the output file without the .png suffix
       image_data: Input numpy array
       Returns:
       Writes PNG file to disk.
       Raises:
       KeyError: Raises an exception.
    """
    im = Image.fromarray(image_data)
    im = im.convert('L')
    im.save(name+".png")
    return

#---------------------

# A modified version of Hongming's crop_center function
def crop_centre(img, cropx, cropy):
    """"
       This function crop images from centre to given size.
       Args:
       img: input image
       cropx: output image width
       cropy: output image height
       Returns:
       data of cropped img
       Raises:
    """
    
    xsize = np.shape(img)[0] # image width
    ysize = np.shape(img)[1] # image height
    startx = xsize//2-(cropx//2)
    starty = ysize//2-(cropy//2)
    img_slice = img[starty:starty+cropy,startx:startx+cropx]
    # This is a sub-optimal solution
    return img_slice

#---------------------

# Nicked from alkasm on stackoverflow
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (np.rint(w/2), np.rint(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

#---------------------

def get_blacklist():
    """Pull up the list of images with known issues"""
    
    blacklist = ['100_036.061+000.563_0.1400_0093.07', '200_003.198+000.788_0.1484_0026.82', '200_033.655+000.710_0.2900_0030.76', '210_348.925-000.435_0.0910_0050.37']
    
    return blacklist

#---------------------

def get_fitsfiles(path):
    """Find all fits files at the path given"""

    filelist = glob.glob(path+'*.fits')
    
    return filelist
    
#---------------------

# A modified version of Hongming's read_fits_image function
def read_fits_image(fitsfile, extension='_F', cropsize=150, pixel_arcsec = 1.8, angularcrop=False):
    """
       This function extracts the image data from a FITS image, clips
       and linearly scales it.
       Args:
       fitsfile: Path to the input FITS file
       Returns:
       image_data: Numpy array containing image from FITS file
       Raises:
       KeyError: Raises an exception.
    """
    
    # Obtaining the naming convention
    namestring = fitsfile[18:52]
    
    with fits.open(fitsfile, ignore_missing_end=True) as hdu:
        image_data = hdu[0].data
        hdu.close()
    image_data = fits.getdata(fitsfile)
    
    a = sigma_clipped_stats(image_data,sigma=3.0, maxiters=5)
    image_data[np.where(image_data<=3*a[2])] = 0.0
    
    image_data = crop_centre(image_data, cropsize, cropsize)
    
    if angularcrop == True:
        
        source_size = float(namestring[27:40]) # Extent of radio source in arcsec
        pixel_radius = np.ceil(np.ceil(source_size/pixel_arcsec)/2) # Converting to source radius in pixels
        
        if pixel_radius > 75:
        
            pixel_radius = 75
            
        mask = create_circular_mask(cropsize, cropsize, radius = pixel_radius)
        
    else:
        
        mask = create_circular_mask(cropsize, cropsize, radius = cropsize/2)
    
    subset = image_data[mask]
    image_data[~mask] = 0
    
    # normalise to [0, 1]:
    image_max, image_min = subset.max(), subset.min()
    image_data[mask] = (image_data[mask] - image_min)/(image_max - image_min)
    # remap to [0, 255] for greyscale:
    image_data*=255.0
    
    # Check if the file has already been saved as the final png
    if path.exists('./MiraBest'+extension+'/PNG/Scaled_Final/'+namestring+'.png') is False:
    
        # Save for a final time here, as 'name.png'
        image_convert('./MiraBest'+extension+'/PNG/Scaled_Final/'+namestring, image_data)
    
    return image_data

#---------------------