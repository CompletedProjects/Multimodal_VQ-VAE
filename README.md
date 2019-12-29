# Multimodal_VQ-VAE
Implementation of a multi-modal VQ-VAE in Pytorch

Based on implementation by unixpickle/vq-vae-2 - https://github.com/unixpickle/vq-vae-2

Dataset used is the FLIR thermal dataset - https://www.flir.co.uk/oem/adas/adas-dataset-form/ which is an autonomous navigation dataset with RGB images and jpeg thermal images. The thermal images are grayscale images used to represent the different temperature values in a scene. The temperature values of the image cannot be obtained to the best of my knowledge due to the calibration information such as the emissivity not being present, which are typically find in rjpeg images ( which are another type of thermal image)
