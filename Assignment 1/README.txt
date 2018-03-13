This project should contain the following files:
	faces.pdf		The report
	faces.tex		The tex file used to generate the report
	faces.py		Main file to be used to generate all results
	get_data.py		Code associated with image processing
	classifier.py		Code associated with classification
	faces.zip		All additional necessary files

faces.zip contains the images used in the report, as well as the cropped 
images. Only the cropped images are included to save space as the 
uncropped images are not used in any place in the code. They, of course,
can still be downloaded with faces.py. faces.zip contains 
actor_image_data.txt, an unmodified but renames version of the dataset
procided in the assignment.

faces.zip also contains the folder _minted-faces. This folder is only
required for compiling the LaTeX document, and then is not strictly
necessary. This program uses the minted package for code styling which
uses the python pygmentize as a lexer to properly highlight syntax in
the code. Normally, this requires running the pdflatex with the
--shell-escape flag set so that it execute python code as part of
compilation. Of course, this raises security concerns on your end,
so I have generated a cache of the results that compilation refers to 
instead. This is what is contained in _minted-faces, provided for your 
convenience.