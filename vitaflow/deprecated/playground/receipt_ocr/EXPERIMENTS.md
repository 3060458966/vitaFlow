## Flow 
As a part of the experiment we run the following steps
- We first seperate the files less than 320x320 pixels in order to reduce the error 
of the character detection algo
- Then pass the images through character detection with threshold 0.6 
- Get the images with characters detected using Advanced East.
- Use the detected co-ordinates of the text to localize the receipt
    - take the top most and bottom most detected text bounding boxes as top and bottom boundaries
    - for left and right lines
        - there is some degree of skewness 
        - this can be solved using average slope of the left and right bounding boxes **in-progress**
- apply perspective transform (use the top left right and bottom cordinates)      
- Fix the rotation 
- Detect regions in the image.
- Detect required information within each region
- OCR fine-grained images
- Present the information in a human-readable format
       
## Problems
- Cannot process images < 320px with Advanced east for text detection
- Need to clean images manually which have watermarks. 
Ideally, assuming this will not be the case always, we keep this case to be tackled at later stage.

## Furture Plans
- Push each and every ML pretrained/out of the box model as container so they can be used for inference purpose independently.