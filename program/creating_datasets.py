# Authors: Sierra Janson
# Affiated with: Brant Robertson
# Date: 05/23/2024 - 07/31/2024

import os
import argparse
import pyvo
import time
import requests
from pyvo.dal.adhoc import DatalinkResults, SodaQuery
from astropy.io import fits
from astropy import nddata
import numpy as np
import matplotlib.pyplot as plt    

# print("YOU ARE USING A BUFFER SINCE EXISTING IMAGE SCRIPT HASN'T BEEN WRITTEN YET")
# BUFFER = 6
FLUX_IMG_SHAPE = (4200,4100)                        


#######################################
# Create command line argument parser
#######################################

def create_parser():

    # Handle user input with argparse
    parser      = argparse.ArgumentParser(
    description = "Detection flags and options from user.")
    
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
    
    # LSST image-retrieval only
    parser.add_argument('--onlyretrieveimages',
        required=False,
        action='store_true',
        help='Specify that user only wants to query and retrieve LSST images, not perform sersic fitting.',
        default=False)
    
    # specify path to write LSST images to
    parser.add_argument('--lsstimagespath',
        dest='lsstimagespath',
        type=str,
        required=False,
        help='Specify that user only wants to query and retrieve LSST images, not perform sersic fitting.',
        default='retrieved_fits')
    
    # specify path to write Sersic outputs of script (including retrieved LSST image) to 
    parser.add_argument('--outputpath',
        dest='outputpath',
        type=str,
        required=False,
        help='Specify path that segmentation map (.npy), Sersic fit values & PSF FITS table, and class-labelled segmentation FITS image are written to.',
        default='sersic_output')

    return parser

#######################################
# authenticate() function
#######################################
def authenticate(tokenfilepath):
        # follow instructions from RSP below to set up the API and retrieve your own token
        # ENSURE you keep your token private
        # https://dp0-2.lsst.io/data-access-analysis-tools/api-intro.html
        RSP_TAP_SERVICE = 'https://data.lsst.cloud/api/tap'
        token_file      = ''

        # if a file path was provided
        try:
            if (tokenfilepath != None): 
                token_file         = tokenfilepath
                
            else:
                homedir            = os.path.expanduser('~')
                secret_file_name   = ".rsp-tap.token" 
                token_file         = os.path.join(homedir,secret_file_name)
                
            with open(token_file, 'r') as f:
                token_str = f.readline()
            
            cred       = pyvo.auth.CredentialStore()
            cred.set_password("x-oauth-basic", token_str)
            credential = cred.get("ivo://ivoa.net/sso#BasicAA")
            service    = pyvo.dal.TAPService(RSP_TAP_SERVICE, credential)
            return service
        
        except FileNotFoundError:
            raise Exception("The path you provided is not recognized by the system.")
            
        except:
            raise Exception("Something failed while attempting to set up the API. This may be because you are not using a Linux system. In which case you will have to alter the 'authenticate()' function in the code yourself, until I adjust this.")
    
    
#######################################
# main() function
#######################################
is_verbose = False
def image_retrieval(args):
    import json
    import datetime
    # begin timer
    time_global_start = time.time()

    # authenticate TAPS
    service = authenticate(args.tokenfilepath)

    ra     = []
    dec    = []
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

        # ensuring filter is valid
        if filter[itera].lower() not in "ugrizy":
            print("The DP0.2 simulation (of whose images this script retrieves) provides images in the u, g, r, i, z, and y bands. This query will be skipped over.")
            print("Please trying again and specify one of the aforementioned filters instead.")
            continue
        
        # configuring query
        # search part
        query = """SELECT TOP %d dataproduct_type,dataproduct_subtype,calib_level,lsst_band,em_min,em_max,lsst_tract,lsst_patch, lsst_filter,lsst_visit,lsst_detector,lsst_ccdvisitid,t_exptime,t_min,t_max,s_ra,s_dec,s_fov, obs_id,obs_collection,o_ucd,facility_name,instrument_name,obs_title,s_region,access_url, access_format FROM ivoa.ObsCore WHERE calib_level = 3 AND dataproduct_type = 'image' AND lsst_band='%s' AND dataproduct_subtype = 'lsst.deepCoadd_calexp'AND CONTAINS(POINT('ICRS', %d, %d), s_region)=1"""%(1, filter[itera].lower(), ra[itera], dec[itera])

        
        # downloading images
        results = service.search(query)

        if (len(results) == 0):
                print(f'No results for query of "RA={ra[itera]}, DEC={dec[itera]}, FILTER={filter[itera]}".')
                print("The DP0.2 simulation (of whose images this script retrieves) covers 300 square degrees centered (RA, DEC) = 61.863, -35.790 degrees. Ensure your RA and DEC are within this boundary.")
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
                        path = args.lsstimagespath
                        if (not os.path.exists(path)):
                                os.mkdir(path)
                                print(f'Path:"{path}" did not exist, but it does now.')
                                
                        if response.status_code == 200:
                                print("Image successfully retrieved.")
                                image_name = f"image{i+BUFFER}_{filter[itera]}_{round(float(ra[itera]))}_{round(float(dec[itera]))}"
                                with open(f"{path}/{image_name}.fits", 'wb') as file:
                                        file.write(response.content)
                                        successful_images.append(f"{path}/{image_name}.fits")
                                retrieved_imgs += 1
                        else:
                                if (args.verbose): print(f"Failed to download file. Status code: {response.status_code}")
                        if (args.verbose): print("Successfully downloaded %d images."%retrieved_imgs)
                        
    if (len(successful_images) == 0):
        if (args.verbose):
            print("No LSST images were successfully retrieved. The script cannot continue, and will exit here.")
        quit()
    #end timer
    time_global_end = time.time()
    if(args.verbose):
            print("Time to retrieve images: %d seconds."%(time_global_end-time_global_start))
    
    with open(f"downloaded_LSST_images.txt", "a") as file:
        img_catalog = open("downloaded_LSST_images.txt","r")
        downloaded_images = img_catalog.readlines()
        
        imagecount = len(downloaded_images)
        for i in successful_images:
            file.write(f"{imagecount}")
            file.write("\t")
            file.write(f"{datetime.datetime.now()}")
            file.write("\t")
            file.write(i)
            file.write("\n")
            imagecount+=1
    
    return successful_images

def set_up(image_path):
        """Returns image, variance, and a graphable PSF from a provided FITS filepath"""
        hdul            = fits.open(image_path)

        image           = hdul[1].data            # image
        variance        = hdul[3].data            # variance
        
        # initializing image shape
        global FLUX_IMG_SHAPE
        FLUX_IMG_SHAPE = image.shape
        
        psfex_info      = hdul[9]
        psfex_data      = hdul[10]
        pixstep         = psfex_info.data._pixstep[0]  # Image pixel per PSF pixel
        size            = psfex_data.data["_size"]  # size of PSF  (nx, ny, n_basis_vectors)
        comp            = psfex_data.data["_comp"]  # PSF basis components
        coeff           = psfex_data.data["coeff"]  # Coefficients modifying each basis vector
        psf_basis_image = comp[0].reshape(*size[0][::-1])
        psf_image       = psf_basis_image * psfex_data.data["basis"][0, :, np.newaxis, np.newaxis]
        psf_image       = psf_image.sum(0)
        psf_image /= psf_image.sum() * pixstep**2
        
        # open template_hdu for writing
        argstemplate = 'morph_cat_template.fits'
        template_hdu = fits.open(argstemplate)
        
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
        return image, variance, psf_image, template_hdu

def identify_sources(image, threshold=1.9):
        """identify sources in fits image by threshold of how bright they are with respect to background RMS"""
        from photutils.background   import Background2D, MedianBackground
        from astropy.convolution    import convolve
        from astropy.convolution    import Gaussian2DKernel
        from photutils.segmentation import detect_sources
        
        # from photutils.segmentation import make_2dgaussian_kernel not comptabile with py 3.7
        # kernel = make_2dgaussian_kernel(3.0, size=5)  # FWHM = 3.0
        
        bkg_estimator    = MedianBackground()
        bkg              = Background2D(image, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
        threshold        = threshold * bkg.background_rms 
        stddev           = 3/(2.35482)
        kernel           = Gaussian2DKernel(stddev, x_size=5, y_size=5)   
        convolved_data   = convolve(image, kernel)
        segment_map      = detect_sources(convolved_data, threshold, npixels=10)
        
        # removing sources within 10 pixels of border to improve quality of training_dataset
        segment_map.remove_border_labels(10, partial_overlap=False, relabel=True)
        return segment_map


# HELPER FUNCTIONS ----------------------#
# DELETE GEN MASK SOON ----------------------------------------------------------------#
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

def gen_psf(image_shape):
    from scipy.ndimage import gaussian_filter
    psf = np.zeros(image_shape)
    center = (image_shape[0] // 2, image_shape[1] // 2)
    psf[center] = 1
    sigma = 2  
    psf = gaussian_filter(psf, sigma=sigma)
    psf /= psf.sum()
    return psf
    

def create_cutouts(segment_map, image, variance, psf):
        """create same dimension cutouts around sources in the image & the variance & psf image"""
        import jax.numpy as jnp

        # grab bounding boxes (slices of cutouts)
        bbox = segment_map.bbox
        # grab source ids
        labels = segment_map.labels
        # initialize mask imgae
        mask_img = np.logical_not(segment_map.data)

        # the number of cutouts and the number of labels should always be the same
        assert(len(bbox) == len(labels))
        
        if(is_verbose): print("Creating cutouts...")
        cutouts = []
        import random
        # for each cutout from detect_sources, cutout a mask image, variance image, and psf image
        for i in range(len(bbox)):
                y_center, x_center = bbox[i].center
                x_len,y_len = bbox[i].shape
                min_length = 12 #22
                
                if (x_len> 10 and y_len > 10 and x_len < 40 and y_len < 40):
                        length        = max([x_len, y_len, min_length]) * 1.25
                        cutout_img    = nddata.Cutout2D(image, (x_center,y_center), int(length))
                        
                        cutout_length = max(cutout_img.shape[0], cutout_img.shape[1])
                        cutout_shape  = (cutout_length,cutout_length)
                    
                        # make mask out of segmentation map cutout
                        xs, ys        = cutout_img.slices_original
                        xmin, xmax    = xs.start, xs.stop
                        ymin, ymax    = ys.start, ys.stop

                        # if the image is not a square
                        if (cutout_img.data.shape[0] != cutout_img.data.shape[1]):
                            
                            def recrop(image, x_len, y_len, length):
                                lchange = length//2
                                rchange = length//2 + length%2
                                if (x_len < y_len): # x < y
                                    if (xmin - lchange < 0 or xmax + rchange >= FLUX_IMG_SHAPE[1]):
                                        difference = y_len - x_len
                                        lchange = difference//2
                                        rchange = difference//2 + difference%2
                                        return image[xmin:xmax, ymin+lchange:ymax-rchange]

                                    return image[xmin-lchange: xmax+rchange, ymin:ymax]

                                else: # dimension == "y" <--> y < x
                                    if (ymin - lchange < 0 or ymax + rchange >= FLUX_IMG_SHAPE[1]):
                                        difference = x_len - y_len
                                        lchange = difference//2
                                        rchange = difference//2 + difference%2
                                        return image[xmin+lchange:xmax-rchange, ymin:ymax]

                                    return image[xmin:xmax, ymin-lchange: ymax+rchange]
                                
                            x_len = cutout_img.data.shape[0]
                            y_len = cutout_img.data.shape[1]
                            y_gt_x = x_len < y_len
                            if (y_gt_x): # if y is greater
                                img_data              = recrop(image, x_len, y_len, cutout_length) 
                                resized_var           = recrop(variance, x_len, y_len, cutout_length)
                                resized_seg_cutout    = recrop(segment_map.data, x_len, y_len, cutout_length)
                                resized_mask          = jnp.array(resized_seg_cutout != labels[i])  # if segment_map == source_id = False (0) else True (1)
                                actual_psf    = resize_image(psf, img_data.shape)
                                assert(resized_mask.shape == img_data.shape)
                                package              = [cutout_img, img_data, resized_mask, resized_var, actual_psf, labels[i]] 
                                cutouts.append(package)
                        else: # cutout is square
                            seg_cutout    = segment_map.data[xmin:xmax,ymin:ymax]
                            cutout_mask   = jnp.array(seg_cutout != labels[i])  # if segment_map == source_id = False (0) else True (1)
                            actual_psf    = resize_image(psf, cutout_shape) # before not reversed
                            cutout_var    = nddata.Cutout2D(variance, (x_center,y_center), int(length))
                            package       = [cutout_img,cutout_img.data, cutout_mask, cutout_var.data, actual_psf,labels[i]] 
                            cutouts.append(package)
        return cutouts

def determine_class(n):
    if (0 < n and n < 1.5):
        return 1
    if (1.5 <= n and n < 3):
        return 2
    if (3 <= n and n < 5):
        return 3
    if (5 <= n):
        return 4
    return 5
    
    
def cutout_sersic_fitting(template_hdu, cutouts):
    """fit sersic profiles to source cutouts using pysersic"""
    from pysersic import FitSingle
    from pysersic.priors import SourceProperties
    from pysersic import check_input_data
    from pysersic import FitSingle
    from pysersic.loss import gaussian_loss
    from pysersic.results import plot_residual
    from jax.random import PRNGKey
    import warnings
    warnings.filterwarnings("ignore")

    
    ##############################################
    # Initializing FITS file for output ---------#
    ##############################################
    # creating fits template
    idxh = {'PRIMARY':0, 'STAT_TABLE':1}

    n_obj = len(cutouts) # number of sources
   
    primary_hdu          = template_hdu[idxh['PRIMARY']]
    stats_template       = template_hdu[idxh['STAT_TABLE']].data
    stats_cat            = fits.FITS_rec.from_columns(stats_template.columns, nrows=n_obj, fill=True)
    
    # initialize segmap FITS image
    layers = [fits.PrimaryHDU()]
    num_classes = 5

    for i in range(num_classes):
        layers.append(fits.ImageHDU(data=np.zeros(FLUX_IMG_SHAPE),name=f"Segmentation Layer {i}"))
    hdul = fits.HDUList(layers)
#     layers.append(np.zeroes(FLUX_IMG_SHAPE))
    
    # initialize markdown for metadata
    markdown = "segmap_info.md" 
    with open(markdown, "w") as md:
        md.write("# Segmentation Map with Sersic Index Classes Description\n\n")
        md.write("Each Image HDU is labelled with either 0, or a Sersic index class # (defined below) from (1-5). These HDUs can be stacked together to create a nuanced image.")
        md.write("\n## Sersic Index Classes:\n\n")
        md.write("Class #1. 0 < n < 1.5\n"
                 "Class #2. 1.5 < n < 3\n"
                 "Class #3. 3 < n < 5\n"
                 "Class #4. n > 5\n"
                 "Class #5. n == 0\n")
    
    
    ##############################################
    # Sersic Fitting Each Cutout ----------------#
    ##############################################
    for i in range(n_obj):
        im,im_data,mask,sig,psf,label = cutouts[i] # image, mask, variance, psf
        if (im_data.shape[0] != im_data.shape[1]):
                print('This should not happen. Pysersic dictates that the cutouts must be square.')
                print(im_data.shape)
        else:
            # try:
            # Prior Estimation of Parameters
            props = SourceProperties(im_data,mask=mask) 
            prior = props.generate_prior('sersic',sky_type='none')

            fitter     = FitSingle(data=im_data,rms=sig, psf=psf, prior=prior, mask=mask, loss_func=gaussian_loss) 
            map_params = fitter.find_MAP(rkey = PRNGKey(1000));                      # contains dictionary of Sersic values

            # can see residual plot of model and flux cutout if desired: 
            # fig, ax = plot_residual(im.data,map_params['model'],mask=mask,vmin=-1,vmax=1);

            ##############################################
            # Testing Fit -------------------------------#
            ##############################################
            image   = im_data
            xc      = map_params["xc"]
            yc      = map_params["yc"]
            flux    = map_params["flux"]
            r_eff   = map_params["r_eff"]
            n       = map_params["n"]
            ellip   = map_params["ellip"]
            theta   = map_params["theta"]
            model   = map_params["model"]
            assert(image.shape == model.shape)

            n_class       = determine_class(n) 

            # retrieving original indices of cutout
            xs, ys        = im.slices_original
            xmin, xmax    = xs.start, xs.stop
            ymin, ymax    = ys.start, ys.stop

            # creating cutout of segmap & labelling with class #
            segmap_cutout = np.logical_not(mask) 
            # segmap_cutout[segmap_cutout == 1] = n_class
            x_coords, y_coords = np.where(segmap_cutout == 1)
            coords = zip(x_coords, y_coords) 
            for (x,y) in coords:
                    if (x+xmin > 4100 or y+ymin>4100): print('yikes')
                    layers[n_class].data[x+xmin,y+ymin] = n_class


            # writing this to specific layer in FITS image 
            # layers[n_class].data[x_min:x_max, y_min:y_max] = segmap_cutout


            # Chi-squared Statistic ----------------------------------------------------------------#
            # (evaluating whether the difference in Image and Model is systematic or due to noise)
            from scipy.stats import chi2

            chi_square           = np.sum((image*2.2 - model) ** 2 / (model))
            df                   = image.size-1                                      # number of categories - 1
            p_value              = chi2.sf(chi_square, df)

            #L1-Norm 
            noise_threshold      = np.mean(sig.data)                               
            image_1D             = image.flatten()
            model_1D             = model.flatten()
            difference_1D        = image_1D - model_1D

            l1                   = np.sum(np.abs(difference_1D))
            l1_normalized        = l1/(image_1D.size)
            l1_var_difference    = l1_normalized - noise_threshold

            ##############################################
            # Creating a FITS image with relevant data --#
            ##############################################
            x,y            = im.center_cutout
            ccols          = [label,x,y] #id, x, y
            morph_params   = [xc, yc, flux, n, r_eff, ellip, theta]
            stats          = [p_value,l1_var_difference]
            values         = ccols + morph_params + stats

            for j in range(len(values)):
                stats_cat[i][j] = values[j]

            # except Exception as error:
            #         print(f"error with image number {i}.")
            #         print(f"Error: {error}")

    template_hdu[idxh['STAT_TABLE']].data = stats_cat
    return template_hdu, hdul
                                
def sersic_fitting_process(args, image_path, i):
        """Manager function for entire process"""
        time_global_start = time.time()
        
        # process:
        image, variance, psf, template_hdu    = set_up(image_path)                                            # grabbing image, variance, psf, and output file template
        segment_map                           = identify_sources(image)                                       # grabbing segmentation map (with source-ids)
        cutouts                               = create_cutouts(segment_map, image, variance, psf)             # create cutouts of flux image, mask, psf, variance
        template_hdu, segmap_hdul             = cutout_sersic_fitting(template_hdu, cutouts)     # fit each cutout with sersic values and save to template_hdu
        
        path = args.outputpath
        
        if (not os.path.exists(path)):
            os.mkdir(path)
            print(f'Path:"{path}" did not exist, but it does now.')

        # writing to disk: 
        np.save(f'{path}/seg{i+BUFFER}.npy',segment_map.data)                        # writing segmentation map (.npy) to disk
        template_hdu.writeto(f'{path}/morph-stats{i+BUFFER}.fits',overwrite=True)    # writing sersic values to disk
        
        segmap_hdul.writeto(f"{path}/labelled_segmap{i+BUFFER}.fits", overwrite=True) # writting labelled segmap to disk

        with fits.open(f'{path}/morph-stats{i+BUFFER}.fits') as hdul:                # open and print stats table for verification
            hdul.info()
            print(hdul[1].data)
        
       
        time_global_end = time.time()
        if(is_verbose): print("Time to perform sersic fitting: %d"%(time_global_end-time_global_start))


#######################################
# Run the program
#######################################
if __name__=="__main__":    
    
    # create the command line argument parser
    parser = create_parser()
    # store the command line arguments
    args = parser.parse_args()
    
    # obtain paths to LSST FITS images
    retrieved_images = image_retrieval(args)    
    
    # if you have to do much testing, it is quicker to use the below in lieu of the above line
    # print("WARNING --------- not retrieving images")
    # retrieved_images = ['retrieved_fits/image0.fits']
    
    if not args.onlyretrieveimages:
        
        # fit each queried image with a sersic profile
        for i in range(len(retrieved_images)):
            sersic_fitting_process(args, retrieved_images[i], i)
                
