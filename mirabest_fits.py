# Okay, so! What we'll need for this one

# All the packages required in the previous making scripts

import numpy as np

from astroquery.skyview import SkyView

import urllib.request
import os.path as path

# A way to provide the main file location's name; should be able to specify a path
# Specification of which survey's being used, and how many pixels the image should be

#---------------------

# Standalone function to put the catalogue list into a useful format
def get_dataset():
    """Opens and formats the list of sources, removing any labelled as unclassifiable or > 270 arcsec in diameter"""
    
    with open('./mirabestdata.txt', 'r') as f:
    data = f.read().splitlines()
    
    dataset = []

    # Splitting out the relevant columns: in order, RA, dec, z, size_rad and FR_class
    for i in range (1, len(data)):
    
        columns = data[i].split()
    
        # A filter to remove any with radial extent greater than the image size
        if float(columns[7]) < 270:
    
        # A filter to remove any class 400 or 402 objects; these are "unclassifiable" and useless for training
            if columns[8] != '400' and columns[8] != '402':
    
                if i == 1:
    
                    dataset = (np.asarray(columns[3:6] + columns[7:9])).astype(np.float)
        
                else:
        
                    columns = (np.asarray(columns[3:6] + columns[7:9])).astype(np.float)
                    dataset = np.concatenate((dataset, columns))

    # Final dataset with arrays of data for individual objects
    dataset = np.reshape(dataset, (-1, 5))
    
    return dataset

#---------------------

# Function to generate the appropriate filename convention for any entry    
def name_string(entry):
    """This takes an entry with columns RA, dec, z, size_rad and class and makes a string to label it"""
    
    label = entry[4].astype(int).astype(str)
    ra = '{:07.3f}'.format(entry[0]*15)  # Moving both into degrees
    dec = '{:+08.3f}'.format(entry[1])  # Retaining sign to keep length consistent
    z = '{:06.4f}'.format(entry[2]) # All redshifts are < 0.5, so keep four significant figures
    rad = '{:07.2f}'.format(entry[3]) # Radial size is a maximum of four figures before point
    
    name = label + '_' + ra + dec + '_' + z + '_' + rad
    
    return name

#---------------------

def image_download(entry, extension='_F', survey='VLA FIRST (1.4 GHz)', pixels=300):
    """Download an image from SkyView using the previously-collected coordinates"""
    
    # Creating the path to the file and the name it'll be saved as
    filename = './MiraBest'+extension+'/FITS/'+name_string(entry)+'.fits'
    
    # Preventing any duplicate downloads
    if path.exists(filename) == False:
    
        coords = (entry[0]*15).astype(str)+', '+entry[1].astype(str)
    
        location=SkyView.get_image_list(position=coords, survey=survey, pixels=pixels)
        
        try:
            
            urllib.request.urlretrieve(location[0], filename)
            print(filename)
            
        except:
            
            print('Problem with url for source at', coords)
            
    return
            
#---------------------

def generate_fits(extension='_F', survey='VLA FIRST (1.4 GHz)', pixels=300):
    """Use all of the previous functions to download all fits files"""
    
    dataset = get_dataset()
    
    for i in range(len(dataset)):
        
        image_download(dataset[i], extension=extension, survey=survey, pixels=pixels)
        
    return
        
#---------------------        