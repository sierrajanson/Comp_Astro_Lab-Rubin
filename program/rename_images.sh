#!/bin/bash

while getopts "hd:f:s:e:t:" opt; do
	case $opt in
		d) images_directory="$OPTARG"
		   	printf 'Will rename all images in provided directory to be prefixed with "save_"\n'
			rename -v $images_directory/ $image_directory/save_ $images_directory/*
		;;
		s) start_index="$OPTARG"
		;;
		e) end_index="$OPTARG"
		;;
		t) images_textfile="$OPTARG"
			# add options to filter based on datetime later

			# rename groups of images by provided index 		 	   	
		 	echo $images_textfile
			if [ -z "$start_index" ] || [ -z "$end_index" ]; then
				printf 'Start index and or end index were not provided; all images listed in text file will be suffixed with "_save".\n'		
				while read index date time filepath; do
					directory="$(dirname "$filepath")"
					filename="$(basename "$filepath")"
					newname="${directory}/save_${filename}"
					mv -v "$filepath" "$newname"
				done < $images_textfile
			else
				printf 'Renaming images from index $start_index to $end_index (not inclusive)...\n'
				while read index date time filepath; do
					if [ "$index" -ge "$start_index" ] && [ "$index" -lt "$end_index" ]; then
						directory="$(dirname "$filepath")"
						filename="$(basename "$filepath")"
						newname="${directory}/save_${filename}"
						mv -v "$filepath" "$newname"
					fi
				done < $images_textfile
				
			fi

		;;
		f) image_path="$OPTARG"
		    	printf 'Will append "_save" to image file name.\n'
	            	rename -v $image_path $image_path*_save. $image_path
		;;
		h)
			printf "Help on options: \n"
			printf "  -d <directory_of_images>       \t provide a directory path of images and every image in the directory will be prefixed with save_\n"
			printf "  -t <text_file_with_imagepaths  \t provide the text file (downloaded_LSST_images.txt) written to disk by creating_dataset.py with information about downloaded images \n"
			printf "  -s <start_index>               \t provide a start index for the textfile (produced by creating_dataset.py option\n"
			printf "  -e <end_index>                 \t provide a end index for the textfile (produced by creating_dataset.py option\n"
			printf "  -f <image_name>		       \t provide the name of an image and it will be suffixed with _save\n"
			printf "  -h                             \t prints this message\n"
			printf "\n"
			printf "About this program: \n"
			printf "  This program provides a variety of options for users to rename image files so they are not overwritten by creating_dataset.py.\n"

		;;
		\?) echo "Invalid option -$OPTARG" >&2
		exit 1
		;;
	esac
done

