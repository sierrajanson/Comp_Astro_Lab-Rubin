# Authors: Sierra Janson
# Affiated with: Brant Robertson
# Date: 05/23/2024

import os
#import matplotlib.pyplot as plt
#from astropy.io import fits
#from astropy.wcs import WCS
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
        secret_file_name = "" 
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
def main():
        #begin timer
        time_global_start = time.time()
        #create the command line argument parser
        parser = create_parser()
        #store the command line arguments
        args = parser.parse_args()

        # authenticate TAPS
        service = authenticate()

        if (args.verbose):
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
                        for i in range(len(fits_images)):
                                 response = requests.get(fits_images[i])
                        retrieved_imgs = 0

                        # checking validity of directory
                        path = 'retrieved_fits'
                        if (not os.path.exists(path)):
                                os.mkdir(path)
                        if response.status_code == 200:
                                with open(f"{path}/test{i}.fits", 'wb') as file:
                                        file.write(response.content)
                                retrieved_imgs += 1
                        else:
                                if (args.verbose): print(f"Failed to download file. Status code: {response.status_code}")
                        if (args.verbose): print("Successfully downloaded %d images."%retrieved_imgs)

        #end timer
        time_global_end = time.time()
        if(args.verbose):
                print("Time to execute program: %d"%(time_global_end-time_global_start))

#######################################
# Run the program
#######################################
if __name__=="__main__":
        main()