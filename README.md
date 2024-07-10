# How to Run
```
cd program
pip install requirements.txt
```
Run using options (band=i, ra=62, dec=-37, verbose=True): 
```
python creating_datasets.py -f i -r 62 -d -37 -v  
```

Run using input file (see provided example file (format --> RA DEC BAND), verbose=True):
```
python creating_datasets.py -n example_input_file.txt -v
```

# Documentation Consulted/Implement Linked Below

https://pysersic.readthedocs.io/en/stable/example-fit.html

https://photutils.readthedocs.io/en/stable/segmentation.html

https://docs.astropy.org/en/stable/api/astropy.nddata.Cutout2D.html
