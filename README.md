# Visual-Jenga-Reimplementation

Reproduction of Visual Jenga (https://visualjenga.github.io/) in PyTorch.


## How to use:

The following command takes in an image filepath and generates a series of images that should naturally take apart the scene in a physically plausible way

`python generate_sequence.py --filename <imagepath> --prompt "point to all books in the image"`

Sample outputs after concatenating images as gifs:

![Description of image](assets/books_output.gif)
![Description of image](assets/cat_output.gif)



The following command takes in an image filepath and generates something similar to figure 2 of the paper:

`python visual_jenga.py --filename <imagepath> --prompt "point to all books in the image"`

![Description of image](assets/inpaint_cat.png)
