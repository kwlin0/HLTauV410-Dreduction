import scipy.ndimage.interpolation as interp
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
import astropy.stats as stat
from photutils import aperture_photometry, CircularAperture, CircularAnnulus, DAOStarFinder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import ZScaleInterval
from astropy.io import fits 
import os 
import glob 
import scipy.ndimage.interpolation as interp
from tqdm import tqdm

def mediancombine(filelist):
    '''
    Takes an input list of files and returns a median combined array of the images in the 
    file list. The returned array is a 2D array with dimensions equal to those of the images/arrays 
    in the file list.

    Args:
        filelist: list of fit files to be combined; each one must have the same dimension


    Returns:
        med_frame: 2D array that is the median combination of the images in filelist. Dimensions are the same as those of the images in filelist
    '''
    # Sets variable n equal to the length of a given file list; n is the number of images
    n = len(filelist)

    # Sets variable first_frame_data equal to the data of the first image in the file list
    first_frame_data = fits.getdata(filelist[0])

    # Uses the first image to find the dimensions of the images in the file list
    # Equates variable imsize_y to the number of values in the y direction (=number of pixels) and
    # imsize_x to the number of values in the x direction
    imsize_y, imsize_x = first_frame_data.shape

    # Initializes a 3D array of zeros with dimensions of the image and a third dimension with size equal to the number
    # of images
    fits_stack = np.zeros((imsize_y, imsize_x , n)) 

    # Inserts the images from filelist into the 3D array fits_stack, making a stack of images
    # the iith image in the stack (index of third dimension) is equal to the iith image in the file list
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im

    # Creates one median (master) frame, which is the median combination of the images in the stack
    med_frame = np.median(fits_stack, axis = 2)

    # Returns median frame
    return med_frame


def filesorter(filename, foldername, fitskeyword_to_check, keyword, current_path = '.'):
    '''
    This function is meant to sort files into folders based on the type of file given (based on keyword). 
    It will create a new folder if one doesn't already exist for a given file type, then make a final check  
    to ensure that the file type in the header information matches the desired type of fit file. The
    files are then placed in their corresponding matching folders.
    
    Args:
    
        filename: the name of the files to be sorted
        foldername: the name of the output folder
        fitskeyword_to_check: the type of file you'd like to sort by as identified in the fits header e.g. BIAS, FLAT
        keyword: the identifier of the keyword to be sorted by e.g. EXPTIME, IMAGETYP
    
    Return:
    
        Files sorted into new directories with the specified folder name
    
    
    '''
    # Check that the given file path exists. If it does, then do nothing, if not, print a warning:
    if os.path.exists(current_path + '/' + filename):
        pass
    else:
       # print(filename + " does not exist or has already been moved.")
        return
    
    header = fits.getheader(current_path + '/' + filename)
    fits_type = header[keyword]
    
    #Check that the given folder path exists. If it does, then do nothing,
    #if not, create a new folder/dir with the give folder name and print that a new directory will be made.

    if os.path.exists(current_path +'/' + foldername):
        pass
    else:
        os.mkdir(current_path + '/' + foldername)

# If the keyword to check matches the fits type according to the header, then put the file
#in the destination, and renames it by the destination and filename. 
    if fits_type.startswith(fitskeyword_to_check):
        if current_path == '.':
            destination = foldername + '/'
            os.rename(current_path +'/' + filename, destination + filename)  
        elif current_path != '.':
            destination = current_path + '/' + foldername + '/'
       # print("Moving " + filename + " to: ./" + destination + filename)
            os.rename(current_path +'/' + filename, destination + filename)  
    return

class Calibration:
    
    class MasterImage:
        def __init__(self, calibration, frames):
            self.calibration = calibration
            self.data = mediancombine(frames)
            self.median = np.median(self.data)
     
            
        def save(self, output_filename, output_dir = os.getcwd()):
            savepath = os.path.join(output_dir,output_filename)
            hdu = fits.PrimaryHDU(self.data)
            hdu.writeto(savepath, overwrite=True)
            print(f"Master {self.calibration.cal_type} file saved to {savepath}")
            
        def show(self):
            plt.figure(figsize=(15,15))
            plt.imshow(self.data, cmap = 'viridis', origin = 'lower')
            return
    
    def __init__(self, cal_type=None):
        
        # Check that a calibration type was provided and that it's 'bias' or 'flats'
        if not cal_type:
            raise Exception("Please provide a calibration type of 'bias' or 'flats'")
        elif cal_type not in ('bias','flats'):
            raise Exception("Unknown calibration type of {cal_type}, please provide a calibration type of 'bias' or 'flats'")
        
        self.cal_type = cal_type
        
        # This set when generate_master() is called
        self.master = None
    
    def __repr__(self):
        representation_text = f"This is a {self.cal_type} calibration.\n"
        
        if self.master is not None:
            representation_text += f"A master {self.cal_type} calibration has been generated\n\n"
            representation_text += "You can now perform one of the following:\n"
            representation_text += ".frames          - retrieve a list of the fits filenames used as calibration frames\n"
            representation_text += ".apply()         - apply this calibration to a fits image\n"
            representation_text += ".master.save()   - save the master calibration to a fits file\n"
            representation_text += ".master.show()   - show a plot of the calibration\n"
            representation_text += ".master.data     - retrieve the data for the master calibration\n"
            representation_text += ".master.median   - get the median of the master calibration image\n"
        else:
            representation_text += "To create a master calibration, use .generate_master(). Call help on the method for more information.\n"

        return representation_text
            
            
    def generate_master_bias(self, bias_frames):
        """Creates a master bias by taking a median combination of all input bias frames.

        Args:
            bias_frames (list of str): a list of all the paths to the bias frames fits files that you'd like to be median combined.

        Example:
            >>> example_calibration = Calibration('bias')
            >>> bias_frames ['image1.fits','image2.fits','image3.fits','image4.fits']
            >>> example_calibration.generate_master(bias_frames)
            >>> example_calibration.master.data
        """
        self.frames = bias_frames
        self.master = self.MasterImage(self, bias_frames)

        return
    
    def generate_master_flat(self, flat_frames):
        '''
        Median combines and normalizes a list of flatfields
        
        Args:
           flat_frames (str): list of flats to be median combined

        Output:
        Returns the median frame
        '''
        self.frames = flat_frames
        self.master = self.MasterImage(self, flat_frames)
        norm = self.master/np.median(self.master)
        
        return
        
        
    def apply(self, input_fits_filepath, save = False, output_dir = None):
        """Apply this calibration to a fits file
        
        Args:
            input_fits_filepath (str): Path to a fits file you want to apply this calibration to
            
        Returns:
            astropy.io.fits.hdu.image.PrimaryHDU - The calibrated image
            
        Example:
            >>> example_calibration = Calibration('bias')
            >>> calibrated_image = example_calibration.apply('myimage.fits')
            
        """
        
       
        if self.master is None:
            raise Exception("In order to use .apply(), you must first generate a master calibration using .generate_master()")

        # Logic here to apply this calibration to the fits file
        # Maybe have something where if it's a bias you divide and if it's a flat you subtract (might be reversed)

        if self.cal_type == 'bias':
            prefix = 'b_'
            def create_calibrated_image(input_fits_filepath):
            
                image = fits.getdata(input_fits_filepath)

                #Calculating the median science_image pixel value
                overscan_median = np.median(image[:,4100:4140])

                #Calculate the ratio to account for drift over night
                drift = overscan_median/self.master.median

                #Scale the master bias by this amount
                master_bias_scaled = self.master.data*drift

                #Bias subtract the science image
                b_subtracted_image = image - master_bias_scaled

                #fits.writeto(output_dir +'b'+ output_filename, b_subtracted_image,fits.getheader(im),overwrite=True)

                hdu = fits.PrimaryHDU(b_subtracted_image)
                return hdu
            
        elif self.cal_type == 'flat':
            prefix = 'f_'
            
            def create_calibrated_image(input_fits_filepath):
            
                data = fits.getdata(input_fits_filepath)


                # Sets variable hdr equal to the header of the input file
                hdr = fits.getheader(input_fits_filepath)

                # Sets variable mf (master flat) equal to the data of the master flat frame
                mf = fits.getdata(path_to_mflat)

                # Creates variable f_data for the flatfielded image; sets it equal to the image array divided by the master flat
                f_data = data/mf

                hdu = fits.PrimaryHDU(b_subtracted_image)
                return hdu
            
        if isinstance(input_fits_filepath,list):
            calibrated_images = []
            
            for image in input_fits_filepath:
                calibrated_image = create_calibrated_image(image) 
                if save:
                    image_filename = image.split('/')[-1]
                    calibrated_image.writeto(os.path.join(output_dir,f'{prefix}{image_filename}'), overwrite=True)
                else:
                    calibrated_images.append(calibrated_image)
           
            if calibrated_images:
                return calibrated_images
            else:
                return


            
            

        return hdu
        
    def overscan_scaling(self, science_image_list, output_dir): 
        """
        Provide an object of type Calibration, and a list of fits files you want to apply that calibration to

        """
        for image in science_image_list:
            calibrated_image = self.apply(image) 
            image_filename = image.split('/')[-1]
            calibrated_image.writeto(os.path.join(output_dir,f'b_{image_filename}'), overwrite=True)

        return
        
        
