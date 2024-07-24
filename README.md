# How to Run
"/sersic_output" will contain the segmentation maps (.npy) and a FITS file with sersic fit information and the PSF (.fits).

"/retrieved_fits" will contain FITS image files retrieved from the Rubin Science platform. You will need your own Rubin API key to retrieve the images. How to provide this information to the script is detailed further below.
```
cd program
pip install requirements.txt
mkdir sersic_output
mkdir retrieved_fits
```
Run using options (band=i, ra=62, dec=-37, verbose=True): 
```
python creating_datasets.py -f i -r 62 -d -37 -v  
```

Run using input JSON file (see example file in /program/):
```
python creating_datasets.py -n example_input.json -v
```

# Acquiring a Rubin API token (mandatory)

[Follow instructions posted by Rubin linked here](https://dp0-2.lsst.io/data-access-analysis-tools/api-intro.html). If you use '.rsp-tap.token' as the token filename as per the tutorial, this is the default path and you need not specify the token file path as a command line option. Otherwise, proceed as below:
```
python creating_datasets.py -n example_input.json -t <tokenfilepath_relative_to_script> -v
```

# About this Program
The script downloads FITS images from LSST based off user-inputted parameters (RA, DEC, and filter), and estimates the Sérsic index and half-light radius of every qualifying source using pysersic. The input paramters can be provided as command line options or through a JSON file. The program outputs the segmentation map of each LSST FITS image, and creates a FITS file with Sérsic values corresponding to a source ID so that each source can be linked to its fit.



# Documentation Consulted/Implement Linked Below

https://pysersic.readthedocs.io/en/stable/example-fit.html

https://photutils.readthedocs.io/en/stable/segmentation.html

https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html
