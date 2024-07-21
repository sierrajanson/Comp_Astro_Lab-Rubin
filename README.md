# How to Run
"/sersic_output" will contain the segmentation maps (.npy) and a FITS file with sersic fit information and the PSF (.fits).
"/retrieved_fits" will contain FITS image files retrieved from the Rubin Science platform. You will need your own Rubin API key to retrieve the images. This will be able to be provided as a command line option soon.
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

# Documentation Consulted/Implement Linked Below

https://pysersic.readthedocs.io/en/stable/example-fit.html

https://photutils.readthedocs.io/en/stable/segmentation.html

https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html
