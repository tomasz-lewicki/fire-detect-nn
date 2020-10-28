# -*- coding: utf-8 -*-
"""4.1.CAM_grad_one_shot(SacramentoTestBurn).ipynb
https://colab.research.google.com/drive/1PcsyaihSRX7px7jdQcskt1XEFiS_RDN6
"""

import os
import datetime
from keras.models import load_model
import keras.backend as K
from keras.preprocessing import image
import numpy as np

import cv2

# Preprocessing

def get_processed_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255. #(the network is trained on 0.0-1.0 inputs)
    return img_tensor


fname = 'blobs/DJI_0003.MOV'
fps = 0.1
score_overlay = False

root_dir = fname.split('.')[0] + f"fps={fps}_created={str(datetime.datetime.now()).replace(' ',':')}" + '/'
if not os.path.isdir(root_dir):
    os.mkdir(root_dir)

frames_dir = root_dir + 'frames/'
if not os.path.isdir(frames_dir):
    os.mkdir(frames_dir)

ffmpeg_return_status = os.system(f"ffmpeg -i {fname} -vf fps={fps} -q:v 1 {frames_dir}/frame%7d.jpg")
print("ffmpeg exited with status: {ffmpeg_return_status}")

hm_dir = root_dir + 'heatmaps/'
if not os.path.isdir(hm_dir):
    os.mkdir(hm_dir)

image_names = os.listdir(frames_dir)
image_names.sort()

print(f'found{len(image_names)} images')

model = load_model('/home/013855803/fire-detect-nn/weights/VGG_monolth_30epochs.h5')

# Build graph for CAM-grad process

binary_fire_output = model.output
last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(model.output, last_conv_layer.output)[0]
print(grads)
pooled_grads = K.mean(grads, axis=(0, 1, 2))
print(pooled_grads)

curr_hm = None

# Run CAM grad process & create heatmaps 
for idx, name in enumerate(image_names):
  
    if idx % 10 == 0: print(idx , "/", len(image_names))
    img_path = frames_dir + name
    img_tensor = get_processed_image(img_path)

    if idx % 10 == 0: 
    # Heatmap generation every 10 frames

        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        #normalize the heatmaps in (0,255)
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        hm_positive = heatmap + np.abs(np.min(heatmap))
        hm_normalized = np.array(255 * hm_positive/np.max(hm_positive), dtype=np.uint8)
        
        #create grayscale image
        img = cv2.imread(img_path)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayimg = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
        
        #apply heatmap
        hm = hm_normalized
        hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

        curr_hm = hm

    superimposed_img = np.uint8(0.3* curr_hm + 0.7* grayimg)
    

    #run inference
    score = model.predict(img_tensor)[0][0]

    #overlay score onto the image
    if score_overlay:
        cv2.putText(superimposed_img, 'flame score: '+str(score),
        (10, 1070), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), thickness=5)
    
    #save image
    img_abs_path = hm_dir+'hm_'+name.split('.')[0]+'.jpg'
    cv2.imwrite(hm_dir+'hm_'+name.split('.')[0]+'.jpg', superimposed_img)

os.system(f"ffmpeg -i {hm_dir}hm_frame%07d.jpg  -codec copy {fname.split('.')[0]}_heatmap_fps={fps}.mp4")

#   #visualize every n images
#   if(idx%100==0):
#     print(idx)
#     f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
#     ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     ax1.set_title('Original ' + name)
#     ax2.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))  
#     ax2.set_title('Image with heatmap')
#     plt.show()

# # ffmpeg conversions (back and forth)
# #!ffmpeg -i '/gdrive/My Drive/fire_dataset/sacramento_test_burn/DJI_0003.MOV' -vf fps=30 -q:v 1 '/gdrive/My Drive/fire_dataset/sacramento_test_burn/DJI_0003_frames/out%5d.jpg'
# #!ffmpeg -i /gdrive/My\ Drive/fire_dataset/sacramento_test_burn/DJI_0003_out/heatmaps/out%05d_heatmap.jpg /gdrive/My\ Drive/fire_dataset/sacramento_test_burn/DJI_0003_out/video_full.mp4

# !ffmpeg -i /gdrive/My\ Drive/fire_dataset/sacramento_test_burn/DJI_0002_out/heatmaps/hm_\ frame%05d.jpg /gdrive/My\ Drive/fire_dataset/sacramento_test_burn/DJI_0002_out/video_full.mp4



