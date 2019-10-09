# PRNet Results

Trying pre-trained out of the box [PRNet](https://github.com/YadiraF/PRNet) for automatically creating detailed face meshes from images. 

It's fantastic at creating an in depth mesh, especially where the face is occluded. In this instance, the face is asymetric in the occluded region. However for VFX editing, e.g. adding horns to a face, as long as the asymmetry is likely low risk.

However, the out of the box model doesn't handle outlier face poses, like a scrunched up face for say an itchy nose:

