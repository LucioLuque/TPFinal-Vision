#  Training dataset. The 91-image dataset is widely used as the training set in learning
# based SR methods [10,5,1]. As deep models generally benefit from big data, studies
#  have found that 91 images are not enough to push a deep model to the best performance.
#  Yang et al. [24] and Schulter et al. [7] use the BSD500 dataset [25]. However, images
#  in the BSD500 are in JPEG format, which are not optimal for the SR task. Therefore,
#  we contribute a new General-100 dataset that contains 100 bmp-format images (with
#  no compression)4. The size of the newly introduced 100 images ranges from 710 704
#  (large) to 131 112 (small). They are all of good quality with clear edges but fewer
#  smooth regions (e.g., sky and ocean), thus are very suitable for the SR training. In
#  the following experiments, apart from using the 91-image dataset for training, we will
#  also evaluate the applicability of the joint set of the General-100 dataset and the 91
# image dataset to train our networks. To make full use of the dataset, we also adopt data
#  augmentation as in [8]. We augment the data in two ways. 1) Scaling: each image is
#  downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with
#  the degree of 90, 180 and 270. Then we will have 5 4 1 = 19 times more images
#  for training.
from PIL import Image

def generate_more_data(images, scale_factors=[0.9, 0.8, 0.7, 0.6], rotations=[90, 180, 270]):
    """
    Generate more training data by scaling and rotating the input images.
    """
    # add scale factor 1 and rotation 0
    scale_factors = [1.0] + scale_factors
    rotations = [0] + rotations

    augmented_images = []
    for img in images:
        for scale in scale_factors:
            for angle in rotations:
                scaled_img = img.resize((int(img.width * scale), int(img.height * scale)), resample=Image.BICUBIC)
                # Rotate the image
                if angle != 0:
                    scaled_img = scaled_img.rotate(angle, expand=True)
                augmented_images.append(scaled_img)
    return augmented_images


# Training samples. To prepare the training data, we first downsample the original train
# ing images by the desired scaling factor n to form the LR images. Then we crop the LR
#  training images into a set of fsub fsub-pixel sub-images with a stride k. The corre
# sponding HR sub-images (with size (nfsub)2) are also cropped from the ground truth
#  images. These LR/HR sub-image pairs are the primary training data.
#  For the issue of padding, we empirically find that padding the input or output maps
#  does little effect on the final performance. Thus we adopt zero padding in all layers
#  according to the filter size. In this way, there is no need to change the sub-image size
#  for different network designs. Another issue affecting the sub-image size is the decon
# volution layer. As we train our models with the Caffe package [27], its deconvolution
#  f
#  ilters will generate the output with size (nfsub n + 1)2 instead of (nfsub)2. So we
#  also crop (n 1)-pixel borders on the HR sub-images. Finally, for 2, 3 and 4, we
#  set the size of LR/HR sub-images to be 102 192, 72 192 and 62 212, respectively.

def prepare_training_samples(images, scale_factor, sub_image_size, stride, border_crop=False):
    """
    Prepare training samples by downsampling and cropping images.
    Scale 2, sub_image_size = (10, 10), stride = 5, border_crop = True.
    Scale 3, sub_image_size = (7, 7), stride = 3, border_crop = True.
    Scale 4, sub_image_size = (6, 6). stride = 2, border_crop = True.
    """
    training_samples = []
    
    for img in images:
        # Downsample the image
        lr_size = (int(img.width // scale_factor), int(img.height // scale_factor))
        lr_img = img.resize(lr_size, resample=Image.BICUBIC)
        
        # Crop LR and HR sub-images
        for y in range(0, lr_img.height - sub_image_size[1] + 1, stride):
            for x in range(0, lr_img.width - sub_image_size[0] + 1, stride):

                lr_patch = lr_img.crop((x, y, x + sub_image_size[0], y + sub_image_size[1]))

                x_hr, y_hr = x * scale_factor, y * scale_factor
                w_hr, h_hr = sub_image_size[0] * scale_factor, sub_image_size[1] * scale_factor
                
                if x_hr + w_hr > img.width or y_hr + h_hr > img.height:
                    continue

                hr_patch = img.crop((x_hr, y_hr, x_hr + w_hr, y_hr + h_hr))

                if border_crop: #if model has deconvolution layer, crop (scale_factor - 1) pixels
                    border = scale_factor - 1
                    if border > 0:
                        hr_patch = hr_patch.crop((border, border,
                                                hr_patch.width - border,
                                                hr_patch.height - border))

                training_samples.append((lr_patch, hr_patch))
    
    return training_samples

def get_dataset(images, scale_factor, sub_image_size, stride, border_crop=False):
    """
    Get the dataset of training samples.
    """
    # Generate more data by scaling and rotating
    augmented_images = generate_more_data(images)

    # Prepare training samples
    training_samples = prepare_training_samples(augmented_images, scale_factor, sub_image_size, stride, border_crop)

    return training_samples