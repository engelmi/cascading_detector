
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from skimage.feature import hog


img = Image.open("C:\\Users\meng\\Desktop\\masterthesis\\svmhog\\tests\\tsne_poiicon_[1,2,3,4,5,6,7,8,9].png").convert("RGB")

image = np.asarray(img)

fd, hog_image = hog(image, orientations=8,
                    pixels_per_cell=(8,8),
                    cells_per_block=(1,1),
                    visualize=True,
                    multichannel=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()


