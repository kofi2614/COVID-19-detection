#!/usr/bin/env python
from keras.models import load_model

model = load_model("DenseNet")





# https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
# import imutils

# import the necessary packages
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
class GradCAM:
	def __init__(self, model, classIdx, layerName=None):
		# store the model, the class index used to measure the class
		# activation map, and the layer to be used when visualizing
		# the class activation map
		self.model = model
		self.classIdx = classIdx
		self.layerName = layerName
		# if the layer name is None, attempt to automatically find
		# the target output layer
		if self.layerName is None:
			self.layerName = self.find_target_layer()
            
	def find_target_layer(self):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(self.model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
	def compute_heatmap(self, image, eps=1e-8):
		# construct our gradient model by supplying (1) the inputs
		# to our pre-trained model, (2) the output of the (presumably)
		# final 4D layer in the network, and (3) the output of the
		# softmax activations from the model
		gradModel = Model(
			inputs=[self.model.inputs],
			outputs=[self.model.get_layer(self.layerName).output,
				self.model.output])
		# record operations for automatic differentiation
		with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
			inputs = tf.cast(image, tf.float32)
			(convOutputs, predictions) = gradModel(inputs)
			loss = predictions[:, self.classIdx]
		# use automatic differentiation to compute the gradients
		grads = tape.gradient(loss, convOutputs)
		# compute the guided gradients
		castConvOutputs = tf.cast(convOutputs > 0, "float32")
		castGrads = tf.cast(grads > 0, "float32")
		guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
		convOutputs = convOutputs[0]
		guidedGrads = guidedGrads[0]
        
		# compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
		weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
		cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
		(w, h) = (image.shape[2], image.shape[1])
		heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
		numer = heatmap - np.min(heatmap)
		denom = (heatmap.max() - heatmap.min()) + eps
		heatmap = numer / denom
		heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
		return heatmap
    
	def overlay_heatmap(self, heatmap, image, alpha=0.5,
		colormap=cv2.COLORMAP_VIRIDIS):
		# apply the supplied color map to the heatmap and then
		# overlay the heatmap on the input image
		heatmap = cv2.applyColorMap(heatmap, colormap)
		output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
		# return a 2-tuple of the color mapped heatmap and the output,
		# overlaid image
		return (heatmap, output)


def get_img_array(img_path): # copied the stuff from above
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    array = image / 255.0
    return array
# convert one record



import matplotlib.pyplot as plt
def plot_heatmap(img_path, model):
    label_dict = {'covid':1,'normal':0,'pneumonia':2}
    label_dict_rev = {v: k for k, v in label_dict.items()}
    orig = cv2.imread(img_path) # orig image
    image = np.reshape(get_img_array(img_path), (1,224,224,3)) # convert from 224,224,3 to 1,224,224,3 to run one imgae
    preds = model.predict(image)[0] # use model to predict
    cam = GradCAM(model, np.argmax(preds[0]))
    heatmap = cam.compute_heatmap(image) # run heat map
    # resize the resulting heatmap to the original input image dimensions
    # and then overlay heatmap on top of the image
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5) # overlay
    print(list(preds))
    print(label_dict_rev[list(preds).index(max(preds))])
    # print(heatmap)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()


plot_heatmap('./dataset\\pneumonia\\00030801_001.png', model)


plot_heatmap('./dataset\\pneumonia\\00000591_004.png', model)


plot_heatmap('./dataset\\covid\\covid-19-caso-91-1-12.png', model)


plot_heatmap('./dataset\\covid\\covid-19-caso-82-1-8.png', model)


plot_heatmap("./dataset\\covid\\1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-001-fig2b.png", model)


plot_heatmap("./dataset\\covid\\1-s2.0-S2387020620301959-gr4_lrg-b.png", model)


plot_heatmap("./dataset\\covid\\1-s2.0-S2387020620301959-gr4_lrg-c.png", model)


plot_heatmap("./dataset\\covid\\1-s2.0-S2214250920300834-gr1_lrg-b.png", model)


plot_heatmap("./dataset\\covid\\10.1016-slash-j.crad.2020.04.002-b.png", model)


plot_heatmap("./dataset\\covid\\23E99E2E-447C-46E5-8EB2-D35D12473C39.png", model)


plot_heatmap('./dataset\\covid\\ryct.2020200028.fig1a.jpeg', model)


plot_heatmap('./dataset\\normal\\00001715_002.png', model)


plot_heatmap('./dataset\\normal\\00001448_000.png', model)


plot_heatmap('./dataset\\normal\\00003191_004.png', model)



