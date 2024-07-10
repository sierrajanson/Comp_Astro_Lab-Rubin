# Authors: Sierra Janson
# Affiated with: Brant Robertson
# Date: 05/23/2024 - 07/09/2024

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
        description="Detection flags and options from user.")

    parser.add_argument('-f', '--filter',
        dest='filter',
        default="i",
        metavar='filter',
        required=False,
        help='Band of desired image.')

    parser.add_argument('-q', '--quantity',
        dest='quantity',
        type=int,
        default=10,
        metavar='filter',
        required=False)

    parser.add_argument('-r', '--ra',
        dest='ra',
        default=62,
        type=int,
        required=False,
        help="Ra of desired image")

    parser.add_argument('-d', '--dec',
        dest='dec',
        default=-37,
        type=int,
        required=False,
        help="Dec of desired image")

    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)

    return parser

#######################################
# authenticate() function
#######################################
def authenticate():
        # follow instructions from RSP below to set up the API and retrieve your own token
        # ENSURE you keep your token private
        # https://dp0-2.lsst.io/data-access-analysis-tools/api-intro.html

        RSP_TAP_SERVICE = 'https://data.lsst.cloud/api/tap'
        homedir = os.path.expanduser('~')
        secret_file_name = "morpheus\\.rubin_secret.txt" 
        assert(secret_file_name != "")
        token_file = os.path.join(homedir,secret_file_name)
        with open(token_file, 'r') as f:
            token_str = f.readline()
        cred = pyvo.auth.CredentialStore()
        cred.set_password("x-oauth-basic", token_str)
        credential = cred.get("ivo://ivoa.net/sso#BasicAA")
        service = pyvo.dal.TAPService(RSP_TAP_SERVICE, credential)
        return service


#######################################
# main() function
#######################################
is_verbose = False
def image_retrieval():
        #begin timer
        time_global_start = time.time()
        #create the command line argument parser
        parser = create_parser()
        #store the command line arguments
        args = parser.parse_args()

        # authenticate TAPS
        service = authenticate()

        if (args.verbose):
                global is_verbose 
                is_verbose  = True
                print("Retrieving %d images with %s filter at (%d,%d)"%(args.quantity,args.filter,args.ra,args.dec))

        # configuring query
        max_image_quantity = 100
        if (args.quantity > max_image_quantity):
                args.quantity = 10
                if (args.verbose): print("Image quantity should not exceed %d"%max_image_quantity)

        # search part
        query = """SELECT TOP %d dataproduct_type,dataproduct_subtype,calib_level,lsst_band,em_min,em_max,lsst_tract,lsst_patch, lsst_filter,lsst_visit,lsst_detector,lsst_ccdvisitid,t_exptime,t_min,t_max,s_ra,s_dec,s_fov, obs_id,obs_collection,o_ucd,facility_name,instrument_name,obs_title,s_region,access_url, access_format FROM ivoa.ObsCore WHERE calib_level = 3 AND dataproduct_type = 'image' AND lsst_band='%s' AND dataproduct_subtype = 'lsst.deepCoadd_calexp'AND CONTAINS(POINT('ICRS', %d, %d), s_region)=1"""%(args.quantity, args.filter,args.ra,args.dec)
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
                        if (args.verbose): print("Downloading %d images..."%len(fits_images))
                        print(fits_images[0])
                        for i in range(len(fits_images)):
                                print("we are in the for loop")
                                response = requests.get(fits_images[i])
                                print("requests executed?")
                        retrieved_imgs = 0
                        print("no for loop???")
                        # checking validity of directory
                        path = 'retrieved_fits'
                        if (not os.path.exists(path)):
                                os.mkdir(path)
                                print("Path did not exist.")

                        if response.status_code == 200:
                                print("Image successfully retrieved.")
                                with open(f"{path}/test{i}.fits", 'wb') as file:
                                        file.write(response.content)
                                retrieved_imgs += 1
                        else:
                                if (args.verbose): print(f"Failed to download file. Status code: {response.status_code}")
                        if (args.verbose): print("Successfully downloaded %d images."%retrieved_imgs)

        #end timer
        time_global_end = time.time()
        if(args.verbose):
                print("Time to retrieve images: %d"%(time_global_end-time_global_start))


from astropy.io import fits
from astropy import nddata
import numpy as np
import matplotlib.pyplot as plt

def set_up(image_path):
        """Returns image, variance, and a graphable PSF from a provided FITS filepath"""
        hdul = fits.open(image_path)
        image = hdul[1].data            # image
        variance = hdul[3].data    # variance
        
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
        from photutils.segmentation import make_2dgaussian_kernel
        from photutils.segmentation import detect_sources
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        threshold = threshold * bkg.background_rms 
        kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        convolved_data = convolve(image, kernel)
        segment_map = detect_sources(convolved_data, threshold, npixels=10)
        segment_map.remove_border_labels(10, partial_overlap=False, relabel=True)
        return segment_map


# HELPER FUNCTIONS ----------------------#
def gen_mask(image_shape):
    """generates mask in the shape of the image"""
    import jax.numpy as jnp
    return jnp.array(np.zeros(image_shape))

def resize_psf(psf_image, new_shape):
        """resizes psf to image size as per specfication of pysersic's FitSingle"""
        import cv2
        resized_psf = cv2.resize(psf_image, new_shape, interpolation=cv2.INTER_AREA)
        resized_psf /= np.sum(resized_psf)    
        return resized_psf

def create_cutouts(segment_map, image, variance, psf):
        """create same dimension cutouts around sources in the image & the variance & psf image"""
        bbox = segment_map.bbox
        labels = segment_map.labels
        assert(len(bbox) == len(labels))

        cutouts = []
        for i in range(len(bbox)):
                y_center, x_center = bbox[i].center
                x_len,y_len = bbox[i].shape
                min_length = 12 #22
                if (x_len> 10 and y_len > 10 and x_len < 40 and y_len < 40):
                        length = max([x_len, y_len, min_length]) * 1.25
                        cutout_img = nddata.Cutout2D(image, (x_center,y_center), int(length))
                        cutout_mask = gen_mask(cutout_img.shape)
                        actual_psf = resize_psf(psf, cutout_img.shape)
                        cutout_var = nddata.Cutout2D(variance, (x_center,y_center), int(length))
                        package = [cutout_img, cutout_mask, cutout_var, actual_psf,labels[i]] #normalized_psf,actual_psf]
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

        labelled_seg = np.zeros((segment_map.shape[0],segment_map.shape[1],3))
        mis_match_count = 0
        for i in range(len(cutouts)):
                im,mask,sig,psf,label = cutouts[i] # image, mask, variance, psf
                if (im.shape[0] != psf.shape[0] or im.shape[1] != psf.shape[1]):
                        print('weird size mismatch for some reason')
                        mis_match_count+=1
                else:
                        check_input_data(im.data, sig.data, psf, mask)

                        # Prior Estimation of Parameters
                        props = SourceProperties(im.data,mask=mask) 
                        prior = props.generate_prior('sersic',sky_type='none')

                        # Fit
                        try:
                                fitter = FitSingle(data=im.data,rms=sig.data, psf=psf, prior=prior, mask=mask, loss_func=gaussian_loss) 
                                map_params = fitter.find_MAP(rkey = PRNGKey(1000));               # contains dictionary of Sersic values
                                # fig, ax = plot_residual(im.data,map_params['model'],mask=mask,vmin=-1,vmax=1);
                                # fig.suptitle("Analysis of Fit")

                                ##############################################
                                # Testing Fit -------------------------------#
                                # (does it belong in training dataset)
                                ##############################################
                                image = im.data
                                model = map_params['model']
                                assert(image.shape == model.shape)

                                # Chi-squared Statistic ----------------------------------------------------------------#
                                # (evaluating whether the difference in Image and Model is systematic or due to noise)

                                from scipy.stats import chi2
                                chi_square = np.sum((image*2.2 - model) ** 2 / (model))
                                df = image.size-1                                                 # number of categories - 1
                                p_value = chi2.sf(chi_square, df)

                                ##############################################
                                # Labelling the Segmap ----------------------#
                                ##############################################
                                n = map_params['n']
                                for xpos in range(im.xmin_original, im.xmax_original,1):
                                        for ypos in range(im.ymin_original, im.ymax_original,1):
                                                if (xpos < image.shape[0] and ypos < image.shape[1]):
                                                        labelled_seg[xpos][ypos] = [label, n, p_value]
                        except Exception as error:
                                print(f"error with image number {i}.")
                                print(f"Error: {error}")
        print(mis_match_count)

                                        



def testing():
       global is_verbose 
       is_verbose = True

def sersic_fitting_process(image_path):
        time_global_start = time.time()
        image, variance, psf = set_up(image_path)
        segment_map = identify_sources(image)
        cutouts = create_cutouts(segment_map, image, variance, psf)
        labelled_segment_map = cutout_sersic_fitting(segment_map, cutouts)
        time_global_end = time.time()
        if(is_verbose):
                print("Time to perform sersic fitting: %d"%(time_global_end-time_global_start))



#######################################
# Run the program
#######################################
if __name__=="__main__":
        # image_retrieval()
        testing()
        sersic_fitting_process('..\\..\\Downloads\\deepCoadd_calexp_3831_3_g_DC2_2_2i_runs_DP0_2_v23_0_1_PREOPS-905_step3_15_20220221T015820Z.fits')