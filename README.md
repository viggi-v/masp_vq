# Codebook Generator for Vector Quantization of Grayscale Images
Requires Numpy, Matplotlib and PIL.
Compresses the image based on codebooks generated using [this research paper](http://ieeexplore.ieee.org/document/4604245/).

`codebook.plot_pca` plots the generated codewords, `codebook.create_codebook` returns the codebooks and blocks used for VQ and `get_compressed_string` returns the blob of the compressed image.
