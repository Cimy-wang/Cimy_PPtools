# Cimy_PPtools
This is a Python toolbox that supports data preprocessing for tasks such as hyperspectral classification and fusion, as well as model saving and result visualization.

If you have any questions, please communicate with me in a timely manner.

The toolbox that includes：
   - * Data Preprocessing * (patching, dimensionality reduction, regularization, etc.),
      - *** createPatches *** [Please be mindful of your development framework.]
            - The input size of Tensorflow and Keras: [B, H, W, C]
            - The input size of Pytorch: [B, C, H, W]
      - *** random_sample ***
      - *** applyPCA ***
      - *** normalization ***
   - * Report Classification Results *,
      - *** reports ***
   - * Training, Validation, Testing *,
      - *** val ***
      - *** train ***
      - *** test ***
   - * Visual Classification Results *！
      - *** label2color ***
      - *** imshow_singlemodal *** If you train a single modal model please use imshow_singlemodal function!
      - *** imshow_multimodal *** If you train a multiple modal model please use imshow_multimodal function!
