# Super Resolution Proba V Kelvins

This repository use a Super Resolution Deep Learning approach in order to solve the PROBA V challenge.
The deep learning models used are : SRCNN and FSRCNN.


To train/validate and test a model, use : DeepLearning-proba-v.ipynb

A baseline using opencv and Bicubic interpolation was also used. It is available in the file : baseline_opencv.ipynb

The best model use the SRCNN concept with few convolution blocks from VGG16. (validation : 0.989 / test 0.984 on leaderboard)
### About the competition
website : https://kelvins.esa.int/proba-v-super-resolution

PROBA-V is an earth observation satellite designed to map land cover and vegetation growth across the entire globe. It was launched on the 6th of May 2013 into a sun-synchronous orbit at an altitude of 820km. It's payload sensors allow for an almost global coverage (90%) per day, providing 300m resolution images. PROBA-V also provides 100m "high resolution" images, but at a lower frequency, of roughly every 5 days (dependent on the location).

The goal of this challenge is to construct such high-resolution images by fusion of the more frequent 300m images. This process, which is known as Multi-image Super-resolution has already been applied to satellite before: some of these satellites, such as SPOT-VGT or ZY-3 TLC, have payloads that are able to take multiple images of the same area during a single satellite pass, which creates a most favorable condition for the use of super resolution algorithms as images are taken simultaneously. Thus, the super resolution can already be included in the final product of these satellites during post-processing on the ground. However, this is not the case for PROBA-V or other satellites, which could benefit from a post-acquisition enhancement. In those cases, multiple images from the same patch are still available, but originate from successive revisits over longer periods of time.

Thus, PROBA-Vs products represent a convenient way to explore super-resolution in a relevant setting. The images provided for this challenge are not artificially degraded, but are real images recorded from the very same scene, just at different resolutions and different times. Any improvements on this data-set might be transferable to larger collections of remote sensing data without the need to deploy more expensive sensors or satellites, as resolution enhancement can happen post-acquisition.