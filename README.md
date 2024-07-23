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


# Documentation Consulted/Implement Linked Below

https://pysersic.readthedocs.io/en/stable/example-fit.html

https://photutils.readthedocs.io/en/stable/segmentation.html

https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html
