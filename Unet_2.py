import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.data_utils import Sequence
import matplotlib.pyplot as plt
import random as random
import matplotlib.animation as animation
import pandas as pd
from functions import *
from tensorflow.keras import Sequential
import sklearn
from sklearn.utils import class_weight
import keras.models
from PIL import Image
import skimage
from keras.callbacks import ModelCheckpoint
import h5py









#Defining constants














movie_duration = 5
depth = 2  #Number of blocks on each side of the Unet, currently 2
level = 1  #Useful if you want to train the Unet output on a intermediate level, not the last one (currently the last one = 1)

digit_max = 10  #Useful if you want to test the training with only a few classes, not all the digits from 0 to 9
max_jump = 1   #Defines what positions are available to a moving digit. If 1, only the closest positions, so 6 directions
max_value = 254
speed = 1   #Defines how far the digit can go at each time step ( x += x + speed * direction )

n_lines = 4  #Just for the plots
n_columns = 3
plot_size = 12

batch_size = 64
nb_batches = 1

movie=True 
n_examples = 3

perturbations = False #whether to add shadows or spotlights to the generate_farames_dataset data
perturbations_time= "random" #if "random", the perturbations can happen anytime. else, int giving the starting frame
perturbations_duration = 2 #How many frames does it last
half= False #whether the shadow or spotlight happens on only half of the frame. if False : whole frame, if True : randomly bottom or top half
spot_proba = 0.5 #probability for the light level variation to be a spotlight or a shadow. 1 for only spotlights, 0 for only shadows
spot_factor= 10 #multiplying factor for the spotlight, usually 10**n
shadow_factor= 0.5 #multiplying factor for the shadow

photo = False   #whether to add the photoreceptor model in front of the Unet
chan1_n = 9 #number of channels, parameter of the PR model

save = True  #whether to save the results in a folder (or in trash)
folder = False  #if False, folder name built from the parameters, if given, string

data_generator = False  #I had errors when I ran it with the data generator, I don't know why

all_epochs_same = True  #Whether all pretraining steps have the same number of epochs (faster than typing all the numbers in the command)
n_epochs=1
step_0, step_1, step_2 = True, True, True

LR = 0.001

plot_params = 0

# Pretraining 0 constants

n_digits_0 = 1
len_train_set_0 = 1000
len_test_set_0 = 100
frame_size_0 = 28
final_size_0 = 4
pool_size_0 = 7
strides_0 = 7
upsample_0 = False
downsample_0 = True

level_0 = level

batch_size_0 = batch_size
nb_batches_0 = nb_batches

n_epochs_0 = 50
LR_0 = LR

# Pretraining 1 constants

n_digits_1 = 1
len_train_set_1 = 1000
len_test_set_1 = 100
frame_size_1 = 28
final_size_1 = 8
pool_size_1 = 7
strides_1 = 3
upsample_1 = False
downsample_1 = True

level_1 = level

batch_size_1 = batch_size
nb_batches_1 = nb_batches

n_epochs_1 =  30
LR_1 = LR


# Training 2 constants

n_digits_2 = 1
len_train_set_2 = 1000
len_test_set_2 = 100
frame_size_2 = 28
final_size_2 = 28
pool_size_2 = 1
strides_2 = 1
upsample_2 = False
downsample_2 = False

level_2 = level

batch_size_2 = batch_size
nb_batches_2 = nb_batches

n_epochs_2 = 20
LR_2 = LR

#To get the variables from the command
read_args(globals())

#Defining the directories and necessary variables
if save:

	if folder != False :
	
		results_directory = folder+"/results/"
		models_directory = folder+"/models/"
		trained_models_directory = folder+"/trained_models/"

		weights_directory = folder+"/weights/"

	else:

		light_name = "shadow"*shadow + "spot"*spot + "nothing"*(not(shadow) and not(spot))
		#shadow_name = "shadow" if shadow else "no_shadow"
		#spot_name = "spot" if spot else "no_spot"
		photo_name = "_photo"*photo
		results_directory = light_name+photo_name+"/results/"

		models_directory = light_name+photo_name+"/models/"
		trained_models_directory = light_name+photo_name+"/trained_models/"

		weights_directory = light_name+photo_name+"/weights/"

else :
    results_directory="trash/"
    models_directory="trash/"
    trained_models_directory="trash/"

    weights_directory = "trash/"
    
if all_epochs_same:
	n_epochs_0, n_epochs_1, n_epochs_2, n_epochs_3, n_epochs_4, n_epochs_5 = n_epochs,n_epochs,n_epochs,n_epochs,n_epochs, n_epochs

if plot_params != 0 :
	#plot_params is a string of shape [param_1,param_2,..]
	list_params = plot_params.replace("[", '')
	list_params = list_params.replace("]", '')
	params = list_params.split(",")
	#params are string, parameters name








#Importing the custom loss
keras.losses.custom_loss = new_weighted_loss

#Importing and normalizing MNIST dataset
(digits_train, labels_train), (digits_test, labels_test) = keras.datasets.mnist.load_data()
digits_train, digits_test = digits_train/255, digits_test/255















if step_0 :
	#Pretraining 0 : downsampled 4x4 MNIST
	
	
	#Generating the dataset

	train_data_0, train_labels_0 = generate_frames_dataset(digits_train, labels_train, 
				                       n_digits=n_digits_0, n_frames=len_train_set_0, 
				                       upsample=upsample_0, frame_size=frame_size_0, 
				                       downsample=downsample_0, 
				                       pool_size=pool_size_0, strides=strides_0, 
				                       final_size=final_size_0, 
				                       movie=movie, duration=movie_duration,
				                       perturbations = perturbations, perturbations_time=perturbations_time, half=half,
		                    			spot_proba = spot_proba, spot_factor=spot_factor, shadow_factor=shadow_factor)

	test_data_0,test_labels_0 = generate_frames_dataset(digits_test, labels_test, 
		                               n_digits=n_digits_0, n_frames=len_test_set_0, 
		                               upsample=upsample_0, frame_size=frame_size_0, 
		                               downsample=downsample_0, pool_size=pool_size_0, strides=strides_0, 
		                               final_size=final_size_0, 
		                               movie=movie, duration=movie_duration,
				                perturbations = perturbations, perturbations_time=perturbations_time, half=half,
		                    		spot_proba = spot_proba, spot_factor=spot_factor, shadow_factor=shadow_factor)




	#Building and compiling the model
	
	
	
	
	
	
	
	unet_model_0 = build_photo_unet_model(final_size_0, movie_duration, photo, digit_max, depth, level_0, chan1_n)

	unet_model_0.summary()
	
	unet_model_0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=[my_accuracy, my_cat_accuracy])



	#To monitor the PR model parameters during training
	checkpoint_0 = ModelCheckpoint(weights_directory+"unet_0_{epoch:02d}.hdf5", save_weights_only=True, monitor='my_acc', verbose=1,save_best_only=False, mode='auto', period=1)
		
	
	
	
	
	
	
	
	#Training (monitoring the parameters only if PR model)

	if photo:
		model_history_0 = unet_model_0.fit(train_data_0, train_labels_0,
				                epochs=n_epochs_0,
				                validation_data=[test_data_0, test_labels_0], 
				                callbacks=[checkpoint_0]
				                )
	else:

		model_history_0 = unet_model_0.fit(train_data_0, train_labels_0,
				                epochs=n_epochs_0,
				                validation_data=[test_data_0, test_labels_0], 
				                )


	
	
	
	
	
	
	
	#Plotting the accuracies, the loss, and if photo, the PR parameters

	my_cat_accuracy_0 = model_history_0.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 0 my_cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("my_cat_accuracy")
	plt.plot(my_cat_accuracy_0)
	plt.savefig(results_directory+"my_cat_accuracy_0")
	plt.close()


	my_accuracy_0 = model_history_0.history["my_accuracy"]
	plt.figure()
	plt.title("step 0 my_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("my_accuracy")
	plt.plot(my_accuracy_0)
	plt.savefig(results_directory+"my_accuracy_0")
	plt.close()


	loss_0 = model_history_0.history["loss"]
	plt.figure()
	plt.title("step 0 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_0)
	plt.savefig(results_directory+"loss_0")
	plt.close()


	if plot_params != 0:
		for param in params :
			param_lists = [[] for k in range(chan1_n)]
			#a list for each channel
			for i in range(1, n_epochs_0+1):
				name = name = (i<10)*str(0) + str(i)
				hf = h5py.File(weights_directory+"unet_0_"+name+".hdf5", 'r')
				w = hf['photoreceptor_da_multichan_randinit']
				w_photo = w['photoreceptor_da_multichan_randinit']
				param_chan = w_photo[param+":0"][0] #param_chan is a list of chan1_n values for the param
				for k, value in enumerate(param_chan):
					param_lists[k].append(value)
					
			plt.figure()
			plt.title(param)
			for k, values_list in enumerate(param_lists):
				plt.plot(values_list, label="chan"+str(k))
			#plt.legend()
			plt.savefig(results_directory+param+"_0")
			plt.close()

	
	
	
	
	
	
	
	#Plotting some example results of the predictions
	
	
	
	predictions_0 = unet_model_0.predict(test_data_0)

	test_movies_0 = [test_data_0[i] for i in range(n_examples)]
	for k, test_movie in enumerate(test_movies_0):

		fig0, ax0 = plt.subplots(1, movie_duration)
		plt.title("the data")
		for i in range(movie_duration):
			ax0[i%movie_duration].imshow(test_movie[:,:,i])
		plt.savefig(results_directory+"data_0_"+str(k))
		plt.close()

		fig1, axs1 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

		for digit in range(digit_max+1):
		    
		    axs1[digit//n_columns,digit%n_columns].set_title(str(digit))
		    axs1[digit//n_columns,digit%n_columns].imshow(test_labels_0[k][:,:,digit])
		plt.savefig(results_directory+"labels_0_"+str(k))

		plt.close()
		    
		fig2, axs2 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

		for digit in range(digit_max+1):
		    
		    prediction_0 = predictions_0[k][:,:,digit]
		    axs2[digit//n_columns,digit%n_columns].set_title(str(digit))
		    axs2[digit//n_columns,digit%n_columns].imshow(prediction_0)
		    
		plt.savefig(results_directory+"predictions_0_"+str(k))

		plt.close()
	
	weights_0 = np.asarray(unet_model_0.get_weights(), dtype="object")
	np.savetxt(weights_directory+"unet_0.csv", weights_0, fmt="%s")
	
	#Saving the models to use the weights for the next training
	unet_model_0.save(trained_models_directory+"unet_0")
















if step_1 :

	#Pretraining 1 : downsampled 8x8 MNIST
	
	
	#Generating the dataset

	train_data_1, train_labels_1 = generate_frames_dataset(digits_train, labels_train, 
				                       n_digits=n_digits_1, n_frames=len_train_set_1, 
				                       upsample=upsample_1, frame_size=frame_size_1, 
				                       downsample=downsample_1, pool_size=pool_size_1, strides=strides_1, 
				                       final_size=final_size_1, 
				                       movie=movie, duration=movie_duration,
				                       perturbations = perturbations, perturbations_time=perturbations_time, half=half,
		                    			spot_proba = spot_proba, spot_factor=spot_factor, shadow_factor=shadow_factor)

	test_data_1,test_labels_1 = generate_frames_dataset(digits_test, labels_test, 
		                               n_digits=n_digits_1, n_frames=len_test_set_1, 
		                               upsample=upsample_1, frame_size=frame_size_1, 
		                               downsample=downsample_1, pool_size=pool_size_1, strides=strides_1, 
		                               final_size=final_size_1, 
		                               movie=movie, duration=movie_duration,
				               perturbations = perturbations, perturbations_time=perturbations_time, half=half,
		                    		spot_proba = spot_proba, spot_factor=spot_factor, shadow_factor=shadow_factor)



	#Building and compiling the model
	
	unet_model_1 = build_photo_unet_model(final_size_1, movie_duration, photo, digit_max, depth, level_1, chan1_n)

	
	unet_model_1.summary()
	
	unet_model_1.load_weights(trained_models_directory+"unet_0",
		                 ).expect_partial()

	unet_model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
		          loss=new_weighted_loss,
		          metrics=[my_accuracy, my_cat_accuracy])


	#Monitoring the parameters
	
	checkpoint_1 = ModelCheckpoint(weights_directory+"unet_1_{epoch:02d}.hdf5", save_weights_only=True, monitor='my_acc', verbose=1,save_best_only=False, mode='auto', period=1)
	
	
	
	
	
	
	#Training 
	
	if photo :

		model_history_1 = unet_model_1.fit(train_data_1, train_labels_1,
				                epochs=n_epochs_1,
				                validation_data=[test_data_1, test_labels_1],
				                callbacks=[checkpoint_1]
				                )
				                
	else :

		model_history_1 = unet_model_1.fit(train_data_1, train_labels_1,
				                epochs=n_epochs_1,
				                validation_data=[test_data_1, test_labels_1],
				                )



	
	
	
	
	
	
	
	
	#Plotting the accuracies, loss, and parameters
	
	
	my_cat_accuracy_1 = model_history_1.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 1 my_cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("my_cat_accuracy")
	plt.plot(my_cat_accuracy_1)
	plt.savefig(results_directory+"my_cat_accuracy_1")
	plt.close()


	my_accuracy_1 = model_history_1.history["my_accuracy"]
	plt.figure()
	plt.title("step 1 my_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("my_accuracy")
	plt.plot(my_accuracy_1)
	plt.savefig(results_directory+"my_accuracy_1")
	plt.close()



	loss_1 = model_history_1.history["loss"]
	plt.figure()
	plt.title("step 1 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_1)
	plt.savefig(results_directory+"loss_1")
	plt.close()


	if plot_params != 0:
		for param in params :
			param_lists = [[] for k in range(chan1_n)]
			#a list for each channel
			for i in range(1, n_epochs_1+1):
				name = name = (i<10)*str(0) + str(i)
				hf = h5py.File(weights_directory+"unet_1_"+name+".hdf5", 'r')
				w = hf['photoreceptor_da_multichan_randinit_1']
				w_photo = w['photoreceptor_da_multichan_randinit_1']
				param_chan = w_photo[param+":0"][0] #param_chan is a list of chan1_n values for the param
				for k, value in enumerate(param_chan):
					param_lists[k].append(value)
					
			plt.figure()
			plt.title(param)
			for k, values_list in enumerate(param_lists):
				plt.plot(values_list, label="chan"+str(k))
			#plt.legend()
			plt.savefig(results_directory+param+"_1")
			plt.close()
	
	
	
	
	
	#Plotting some examples of predictions
	
	
	predictions_1 = unet_model_1.predict(test_data_1)


	test_movies_1 = [test_data_1[i] for i in range(n_examples)]
	for k, test_movie in enumerate(test_movies_1):

		fig3, ax3 = plt.subplots(1, movie_duration)
		plt.title("the data")
		for i in range(movie_duration):
			ax3[i%movie_duration].imshow(test_movie[:,:,i])
		plt.savefig(results_directory+"data_1_"+str(k))

		plt.close()

		fig4, axs4 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

		for digit in range(digit_max+1):
		    
		    axs4[digit//n_columns,digit%n_columns].set_title(str(digit))
		    axs4[digit//n_columns,digit%n_columns].imshow(test_labels_1[k][:,:,digit])
		plt.savefig(results_directory+"labels_1_"+str(k))

		plt.close()

		fig5, axs5 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

		for digit in range(digit_max+1):
		    
		    prediction_1 = predictions_1[k][:,:,digit]
		    axs5[digit//n_columns,digit%n_columns].set_title(str(digit))
		    axs5[digit//n_columns,digit%n_columns].imshow(prediction_1)
		    
		plt.savefig(results_directory+"predictions_1_"+str(k))

		plt.close()
		
	weights_1 = np.asarray(unet_model_1.get_weights(), dtype="object")
	np.savetxt(weights_directory+"unet_1.csv", weights_1, fmt="%s")

	unet_model_1.save(trained_models_directory+"unet_1")

	
	
	
	
	
	
	
	
	
	
	
	
	
if step_2 :

	#Training 2 : MNIST (28x28)
	
	
	#Generating the dataset


	train_data_2, train_labels_2 = generate_frames_dataset(digits_train, labels_train, 
		                               n_digits=n_digits_2, n_frames=len_train_set_2, 
		                               upsample=upsample_2, frame_size=frame_size_2, 
		                               downsample=downsample_2, pool_size=pool_size_2, strides=strides_2, 
		                               final_size=final_size_2, 
		                               movie=movie, duration=movie_duration,
				               perturbations = perturbations, perturbations_time=perturbations_time, half=half,
		                    		spot_proba = spot_proba, spot_factor=spot_factor, shadow_factor=shadow_factor)

	test_data_2,test_labels_2 = generate_frames_dataset(digits_test, labels_test, 
		                               n_digits=n_digits_2, n_frames=len_test_set_2, 
		                               upsample=upsample_2, frame_size=frame_size_2, 
		                               downsample=downsample_2, pool_size=pool_size_2, strides=strides_2, 
		                               final_size=final_size_2, 
		                               movie=movie, duration=movie_duration,
				               perturbations = perturbations, perturbations_time=perturbations_time, half=half,
		                    		spot_proba = spot_proba, spot_factor=spot_factor, shadow_factor=shadow_factor)


	
	
	
	
	#Building and compiling the model
	
	unet_model_2 = build_photo_unet_model(final_size_2, movie_duration, photo, digit_max, depth, level_2, chan1_n)

	unet_model_2.summary()
	
	unet_model_2.load_weights(trained_models_directory+"unet_1",
		                 ).expect_partial()

	unet_model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=[my_accuracy,my_cat_accuracy])


	#Monitoring the PR parameters
	
	checkpoint_2 = ModelCheckpoint(weights_directory+"unet_2_{epoch:02d}.hdf5", save_weights_only=True, monitor='my_acc', verbose=1,save_best_only=False, mode='auto', period=1)
	
	
	
	
	
	
	#Training 
	
	if photo :

		model_history_2 = unet_model_2.fit(train_data_2, train_labels_2,
				                epochs=n_epochs_2,
				                validation_data=[test_data_2, test_labels_2],
				                callbacks=[checkpoint_2]
				                )
				                
	else :

		model_history_2 = unet_model_2.fit(train_data_2, train_labels_2,
				                epochs=n_epochs_2,
				                validation_data=[test_data_2, test_labels_2],
				                )


	
	
	
	
	#Plotting the accuracies, the loss, and the PR parameters

	my_cat_accuracy_2 = model_history_2.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 2 my_cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("my_cat_accuracy")
	plt.plot(my_cat_accuracy_2)
	plt.savefig(results_directory+"my_cat_accuracy_2")
	plt.close()

	my_accuracy_2 = model_history_2.history["my_accuracy"]
	plt.figure()
	plt.title("step 2 my_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("my_accuracy")
	plt.plot(my_accuracy_2)
	plt.savefig(results_directory+"my_accuracy_2")
	plt.close()

	loss_2 = model_history_2.history["loss"]
	plt.figure()
	plt.title("step 2 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_2)
	plt.savefig(results_directory+"loss_2")
	plt.close()

	if plot_params != 0:
		for param in params :
			param_lists = [[] for k in range(chan1_n)]
			#a list for each channel
			for i in range(1, n_epochs_2+1):
				name = name = (i<10)*str(0) + str(i)
				hf = h5py.File(weights_directory+"unet_2_"+name+".hdf5", 'r')
				w = hf['photoreceptor_da_multichan_randinit_2']
				w_photo = w['photoreceptor_da_multichan_randinit_2']
				param_chan = w_photo[param+":0"][0] #param_chan is a list of chan1_n values for the param
				for k, value in enumerate(param_chan):
					param_lists[k].append(value)
					
			plt.figure()
			plt.title(param+"")
			for k, values_list in enumerate(param_lists):
				plt.plot(values_list, label="chan"+str(k))
			#plt.legend()
			plt.savefig(results_directory+param+"_2")
			plt.close()
	
	
	
	#Plotting some example predictions
	
	predictions_2 = unet_model_2.predict(test_data_2)

	test_movies_2 = [test_data_2[i] for i in range(n_examples)]
	for k, test_movie in enumerate(test_movies_2):

		fig6, ax6 = plt.subplots(1, movie_duration)
		plt.title("the data")
		for i in range(movie_duration):
			ax6[i%movie_duration].imshow(test_movie[:,:,i])

		plt.savefig(results_directory+"data_2_"+str(k))

		plt.close()



		fig7, axs7 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

		for digit in range(digit_max+1):
		    
		    axs7[digit//n_columns,digit%n_columns].set_title(str(digit))
		    axs7[digit//n_columns,digit%n_columns].imshow(test_labels_2[k][:,:,digit])
		plt.savefig(results_directory+"labels_2_"+str(k))

		plt.close()

		fig8, axs8 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

		for digit in range(digit_max+1):
		    
		    prediction_2 = predictions_2[k][:,:,digit]
		    axs8[digit//n_columns,digit%n_columns].set_title(str(digit))
		    axs8[digit//n_columns,digit%n_columns].imshow(prediction_2)
		plt.savefig(results_directory+"predictions_2_"+str(k))

		plt.close()

	weights_2 = np.asarray(unet_model_2.get_weights(), dtype="object")
	np.savetxt(weights_directory+"unet_2.csv", weights_2, fmt="%s")

	unet_model_2.save(trained_models_directory+"unet_2")

