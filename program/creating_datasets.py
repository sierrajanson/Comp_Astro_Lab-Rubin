# Authors: Sierra Janson
# Affiated with: Brant Robertson
# Date: 05/23/2024 - 07/23/2024

import os
import argparse
import pyvo
import time
import requests
from pyvo.dal.adhoc import DatalinkResults, SodaQuery

#######################################
# Create command line argument parser
#######################################

def create_parser():

        # Handle user input with argparse
    parser = argparse.ArgumentParser(
    description            = "Detection flags and options from user.")
    
    # input file name (with ra, dec, filter) argument
    parser.add_argument(
        '-n', 
        '--filename',
        dest='filename',
        type=str,
        default=None,
        metavar='filter',
        required=False
    )
    
    # specify filter argument
    parser.add_argument(
        '-f', '--filter',
        dest='filter',
        default="i",
        metavar='filter',
        required=False,
        help='Band of desired image.'
    )

    # specify ra argument
    parser.add_argument('-r', '--ra',
        dest='ra',
        default=62,
        type=int,
        required=False,
        help="Ra of desired image")

    # specify dec argument
    parser.add_argument('-d', '--dec',
        dest='dec',
        default=-37,
        type=int,
        required=False,
        help="Dec of desired image")

    # specify whether helpful information should be outputted
    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)
    
    # specify path to file with Rubin API token
    parser.add_argument('-t', '--tokenfilepath',
        dest='tokenfilepath',
        type=str,
        required=False,
        help='Provide path to file with Rubin API token',
        default=None)

    return parser

#######################################
# authenticate() function
#######################################
def authenticate(tokenfilepath):
        # follow instructions from RSP below to set up the API and retrieve your own token
        # ENSURE you keep your token private
        # https://dp0-2.lsst.io/data-access-analysis-tools/api-intro.html
        RSP_TAP_SERVICE = 'https://data.lsst.cloud/api/tap'
        token_file = ''

        # if a file path was provided
        try:
            if (tokenfilepath != None): 
                token_file = tokenfilepath
            else:
                homedir = os.path.expanduser('~')
                secret_file_name = ".rsp-tap.token" 
                token_file = os.path.join(homedir,secret_file_name)
            with open(token_file, 'r') as f:
                token_str = f.readline()
            cred = pyvo.auth.CredentialStore()
            cred.set_password("x-oauth-basic", token_str)
            credential = cred.get("ivo://ivoa.net/sso#BasicAA")
            service = pyvo.dal.TAPService(RSP_TAP_SERVICE, credential)
            return service
        except FileNotFoundError:
            raise Exception("The path you provided is not recognized by the system.")
        except:
            raise Exception("Something failed while attempting to set up the API. This may be because you are not using a Linux system. In which case you will have to alter the 'authenticate()' function in the code yourself, until I adjust this.")

from astropy.io import fits
from astropy import nddata
import numpy as np
import matplotlib.pyplot as plt    
    
argstemplate = 'morph_cat_template.fits'
template_hdu = fits.open(argstemplate)
    
    
#######################################
# main() function
#######################################
is_verbose = False
def image_retrieval():
    import json
    # begin timer
    time_global_start = time.time()
    # create the command line argument parser
    parser = create_parser()
    # store the command line arguments
    args = parser.parse_args()

    # authenticate TAPS
    service = authenticate(args.tokenfilepath)

    ra = []
    dec = []
    filter = []

    iterations = 0
    if (args.filename != None):
            try:
                print("Opening provided file...")
                with open(args.filename, 'r') as file:
                    data = json.load(file)
                    input_list = data["values"]
                    for element in input_list:
                        ra.append(float(element["ra"]))
                        dec.append(float(element["dec"]))
                        filter.append(element["filter"])
                        iterations+=1
            except FileNotFoundError:
                print(f"Error: The file '{args.filename}' does not exist.")
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        ra.append(args.ra)
        dec.append(args.dec)
        filter.append(args.filter)
        iterations = 1

    successful_images = []
    print(f"Will do {iterations} image retrievals.")
    for itera in range(iterations):
        if (args.verbose):
                global is_verbose 
                is_verbose  = True
                print("Retrieving an image with %s filter at (%d,%d)"%(filter[itera],ra[itera],dec[itera]))

        # configuring query
        # search part
        query = """SELECT TOP %d dataproduct_type,dataproduct_subtype,calib_level,lsst_band,em_min,em_max,lsst_tract,lsst_patch, lsst_filter,lsst_visit,lsst_detector,lsst_ccdvisitid,t_exptime,t_min,t_max,s_ra,s_dec,s_fov, obs_id,obs_collection,o_ucd,facility_name,instrument_name,obs_title,s_region,access_url, access_format FROM ivoa.ObsCore WHERE calib_level = 3 AND dataproduct_type = 'image' AND lsst_band='%s' AND dataproduct_subtype = 'lsst.deepCoadd_calexp'AND CONTAINS(POINT('ICRS', %d, %d), s_region)=1"""%(1, filter[itera], ra[itera], dec[itera])
        #print(query)

        # downloading images
        results = service.search(query)

        if (len(results) == 0):
                if (args.verbose): print("No results for that search.")
        else:
                fits_images = []
                for i in range(len(results)):
                    dataLinkUrl = results[i].getdataurl()
                    auth_session = service._session
                    dl = DatalinkResults.from_result_url(dataLinkUrl, session=auth_session)
                    fits_image_url = dl.__getitem__("access_url")[0]
                    fits_images.append(fits_image_url)
                if (len(fits_images) == 0):
                        if(args.verbose):
                                print("No images retrieved. Possibly error during retrieval")
                else:
                        # retrieve & download images
                        for i in range(len(fits_images)):
                                response = requests.get(fits_images[i])
                        retrieved_imgs = 0
                        # checking validity of directory
                        path = 'retrieved_fits'
                        if (not os.path.exists(path)):
                                os.mkdir(path)
                                print("Path did not exist.")

                        if response.status_code == 200:
                                print("Image successfully retrieved.")
                                with open(f"{path}/image{i}.fits", 'wb') as file:
                                        file.write(response.content)
                                        successful_images.append(f"{path}/image{itera}.fits")
                                retrieved_imgs += 1
                        else:
                                if (args.verbose): print(f"Failed to download file. Status code: {response.status_code}")
                        if (args.verbose): print("Successfully downloaded %d images."%retrieved_imgs)

    #end timer
    time_global_end = time.time()
    if(args.verbose):
            print("Time to retrieve images: %d seconds."%(time_global_end-time_global_start))

    print(successful_images)
    return successful_images

def set_up(image_path):
        """Returns image, variance, and a graphable PSF from a provided FITS filepath"""
        hdul = fits.open(image_path)
        image = hdul[1].data            # image
        variance = hdul[3].data    # variance
        print(image.shape)
        psfex_info = hdul[9]
        psfex_data = hdul[10]
        pixstep = psfex_info.data._pixstep[0]  # Image pixel per PSF pixel
        size = psfex_data.data["_size"]  # size of PSF  (nx, ny, n_basis_vectors)
        comp = psfex_data.data["_comp"]  # PSF basis components
        coeff = psfex_data.data["coeff"]  # Coefficients modifying each basis vector
        psf_basis_image = comp[0].reshape(*size[0][::-1])
        psf_image = psf_basis_image * psfex_data.data["basis"][0, :, np.newaxis, np.newaxis]
        psf_image = psf_image.sum(0)
        psf_image /= psf_image.sum() * pixstep**2
        
        # Appending PSF to FITS output
        image_hdu2 = fits.ImageHDU(data=psf_image, name="PSF")
        template_hdu.append(image_hdu2)
        
        if (is_verbose):
                # Plotting Retrieved FITS Image
                plt.imshow(image.data, vmin=0, vmax=0.3,origin="lower", cmap="gray")
                plt.title('Original Image Array')
                plt.show()

                # Plotting PSF
                plt.imshow(psf_image, cmap='gray', interpolation='none',vmin=-0.0001, vmax=0.0001)
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.show()
        return image, variance, psf_image

def identify_sources(image, threshold=1.9):
        """identify sources in fits image by threshold of how bright they are with respect to background RMS"""
        from photutils.background import Background2D, MedianBackground
        from astropy.convolution import convolve
        # from photutils.segmentation import make_2dgaussian_kernel not comptabile with py 3.7
        from astropy.convolution import Gaussian2DKernel

        from photutils.segmentation import detect_sources
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        threshold = threshold * bkg.background_rms 
        stddev = 3/(2.35482)
        kernel = Gaussian2DKernel(stddev, x_size=5, y_size=5)
        # kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        convolved_data = convolve(image, kernel)
        segment_map = detect_sources(convolved_data, threshold, npixels=10)
        segment_map.remove_border_labels(10, partial_overlap=False, relabel=True)
        return segment_map


# HELPER FUNCTIONS ----------------------#
def gen_mask(image_shape):
    """generates mask in the shape of the image"""
    import jax.numpy as jnp
    return jnp.array(np.zeros(image_shape))

def resize_image(psf_image, new_shape):
        """resizes psf (or any passed image) to image size as per specfication of pysersic's FitSingle"""
        import cv2
        resized_psf = cv2.resize(psf_image, new_shape, interpolation=cv2.INTER_AREA)
        resized_psf /= np.sum(resized_psf)    
        return resized_psf

def create_cutouts(segment_map, image, variance, psf):
        """create same dimension cutouts around sources in the image & the variance & psf image"""
        bbox = segment_map.bbox
        labels = segment_map.labels
        assert(len(bbox) == len(labels))
        if(is_verbose): print("Creating cutouts...")
        cutouts = []
        for i in range(len(bbox)):
                y_center, x_center = bbox[i].center
                x_len,y_len = bbox[i].shape
                min_length = 12 #22
                
                if (x_len> 10 and y_len > 10 and x_len < 40 and y_len < 40):
                        length = max([x_len, y_len, min_length]) * 1.25
                        cutout_img = nddata.Cutout2D(image, (x_center,y_center), int(length))
                        cutout_shape = (cutout_img.shape[0],cutout_img.shape[0])
                        cutout_mask = gen_mask(cutout_shape)
                        actual_psf = resize_image(psf, cutout_shape)
                        cutout_var = nddata.Cutout2D(variance, (x_center,y_center), int(length))
                        package = [cutout_img,cutout_img.data, cutout_mask, cutout_var.data, actual_psf,labels[i]] #normalized_psf,actual_psf]

                        # if not a square
                        if (cutout_img.data.shape[0] != cutout_img.data.shape[1]):
                            resized_cutout = resize_image(cutout_img.data, cutout_shape)
                            resized_var = resize_image(cutout_var.data, cutout_shape)
                            package = [cutout_img, resized_cutout, cutout_mask, resized_var, actual_psf, labels[i]] 
                        cutouts.append(package)
        return cutouts

def cutout_sersic_fitting(segment_map, cutouts):
    """fit sersic profiles to source cutouts using pysersic"""
    from pysersic import FitSingle
    from pysersic.priors import SourceProperties
    from pysersic import check_input_data
    from pysersic import FitSingle
    from pysersic.loss import gaussian_loss
    from pysersic.results import plot_residual
    from jax.random import PRNGKey

    # labelled_seg = np.zeros((segment_map.shape[0],segment_map.shape[1],3))
    mis_match_count = 0 # delete soon
    
    ##############################################
    # Initializing FITS file for output ---------#
    ##############################################
    # creating fits template
    idxh = {'PRIMARY':0, 'STAT_TABLE':1}

    n_obj = len(cutouts) # number of sources
    primary_hdu          = template_hdu[idxh['PRIMARY']]
    stats_template       = template_hdu[idxh['STAT_TABLE']].data
    stats_cat            = fits.FITS_rec.from_columns(stats_template.columns, nrows=n_obj, fill=True)
    
    ##############################################
    # Sersic Fitting Each Cutout ----------------#
    ##############################################
    for i in range(n_obj):
        im,im_data,mask,sig,psf,label = cutouts[i] # image, mask, variance, psf
        if (im_data.shape[0] != im_data.shape[1]):
                print('This should not happen.')
                print(im.data.shape)
                mis_match_count+=1
        else:
                check_input_data(im_data, sig, psf, mask)
                # Prior Estimation of Parameters
                props = SourceProperties(im_data,mask=mask) 
                prior = props.generate_prior('sersic',sky_type='none')

                # Fit
                try:
                    fitter = FitSingle(data=im_data,rms=sig, psf=psf, prior=prior, mask=mask, loss_func=gaussian_loss) 
                    map_params = fitter.find_MAP(rkey = PRNGKey(1000));               # contains dictionary of Sersic values
                    # fig, ax = plot_residual(im.data,map_params['model'],mask=mask,vmin=-1,vmax=1);
                    # fig.suptitle("Analysis of Fit")

                    ##############################################
                    # Testing Fit -------------------------------#
                    ##############################################
                    image = im_data
                    #xc,yc,flux,r_eff,n,ellip,theta,model = map_params
                    xc = map_params["xc"]
                    yc = map_params["yc"]
                    flux = map_params["flux"]
                    r_eff = map_params["r_eff"]
                    n = map_params["n"]
                    ellip = map_params["ellip"]
                    theta = map_params["theta"]
                    model = map_params["model"]
                    assert(image.shape == model.shape)

                    # Chi-squared Statistic ----------------------------------------------------------------#
                    # (evaluating whether the difference in Image and Model is systematic or due to noise)
                    from scipy.stats import chi2

                    chi_square           = np.sum((image*2.2 - model) ** 2 / (model))
                    df                   = image.size-1                                                 # number of categories - 1
                    p_value              = chi2.sf(chi_square, df)

                    #L1-Norm 
                    noise_threshold      = np.mean(sig.data)                               
                    image_1D             = image.flatten()
                    model_1D             = model.flatten()
                    difference_1D        = image_1D - model_1D

                    l1 = np.sum(np.abs(difference_1D))
                    l1_normalized        = l1/(image_1D.size)
                    l1_var_difference    = l1_normalized - noise_threshold
                    print(f"L1 norm: {l1_normalized}")

                    ##############################################
                    # Creating a FITS image with relevant data --#
                    ##############################################
                    x,y            = im.center_cutout
                    ccols          = [label,x,y] #id, x, y
                    morph_params = [xc, yc, flux, n, r_eff, ellip, theta]
                    # morph_params   = [float(xc),float(yc),float(flux),float(n),float(r_eff),float(ellip),float(theta)]
                    stats          = [p_value,l1_var_difference]
                    values         = ccols + morph_params + stats

                    for j in range(len(values)):
                        stats_cat[i][j] = values[j]

                except Exception as error:
                        print(f"error with image number {i}.")
                        print(f"Error: {error}")

    template_hdu[idxh['STAT_TABLE']].data = stats_cat
    return template_hdu
                                
def sersic_fitting_process(image_path, i):
        time_global_start         = time.time()
        image, variance, psf      = set_up(image_path)
        segment_map               = identify_sources(image)
        cutouts                   = create_cutouts(segment_map, image, variance, psf)
        template_hdu              = cutout_sersic_fitting(segment_map, cutouts)
        np.save(f'sersic_output/seg{i}.npy',segment_map.data)
        template_hdu.writeto(f'sersic_output/morph-stats{i}.fits',overwrite=True)
        with fits.open(f'sersic_output/morph-stats{i}.fits') as hdul:
            hdul.info()
            print(hdul[1].data)
        
        time_global_end           = time.time()
        if(is_verbose): print("Time to perform sersic fitting: %d"%(time_global_end-time_global_start))


#######################################
# Run the program
#######################################
if __name__=="__main__":
    retrieved_images = image_retrieval()
    # retrieved_images = ["retrieved_fits/image0.fits"]
    labelled_segs = []
    for i in range(len(retrieved_images)):
        sersic_fitting_process(retrieved_images[i], i)
                
