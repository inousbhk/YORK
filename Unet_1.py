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







#Defining constants









#General constants

movie_duration = 5
depth = 2  #Number of blocks on each side of the Unet, currently 2
level = 1  #Useful if you want to train the Unet output on a intermediate level, not the last one (currently the last one = 1)
LR = 10**(-2)  #learning rate

digit_max = 10  #Useful if you want to test the training with only a few classes, not all the digits from 0 to 9
max_jump = 1   #Defines what positions are available to a moving digit. If 1, only the closest positions, so 6 directions
max_value = 254 
speed = 1   #Defines how far the digit can go at each time step ( x += x + speed * direction )

n_lines = 4  #Just for the plots
n_columns = 3
plot_size = 12

batch_size = 64
nb_batches = 1

shadow = False #whether to add a shadow to the generate_movie_dataset_5 data
shadow_ratio = 0.2  #ratio of shadow surface/total surface
light_intensity = 0.5  #multiplying factor for the shadow

photo = False  #whether to add the photoreceptor model in front of the Unet
chan1_n = 9 #number of channels, parameter of the PR model

save = True  #whether to save the results in a folder (or in trash)
folder=False  #if False, folder name built from the parameters, if given, string

data_generator = False  #I had errors when I ran it with the data generator, I don't know why

all_epochs_same = True  #Whether all pretraining steps have the same number of epochs (faster than typing all the numbers in the command)
n_epochs = 1

step_0, step_1, step_2,step_3, step_4, step_5 = True, True, True, True, True, True #Whether to execute all steps

#Pretraining 0 constants

n_digits_0 = 1
len_train_set_0 = 1000
len_test_set_0 = 100
frame_size_0 = 28
final_size_0 = 4
pool_size_0 = 7
strides_0 = 7
upsample_0 = False
downsample_0 = True
movie_0 = True

level_0 = level

batch_size_0 = batch_size
nb_batches_0 = nb_batches

n_epochs_0 = 50
LR_0 = LR

#Pretraining 1 constants

n_digits_1 = 1
len_train_set_1 = 1000
len_test_set_1 = 100
frame_size_1 = 28
final_size_1 = 8
pool_size_1 = 7
strides_1 = 3
upsample_1 = False
downsample_1 = True
movie_1 = True

level_1 = level

batch_size_1 = batch_size
nb_batches_1 = nb_batches

n_epochs_1 =  30
LR_1 = LR


#Pretraining 2 constants

n_digits_2 = 1
len_train_set_2 = 1000
len_test_set_2 = 100
frame_size_2 = 28
final_size_2 = 28
pool_size_2 = 1
strides_2 = 1
upsample_2 = False
downsample_2 = False
movie_2 = True

level_2 = level

batch_size_2 = batch_size
nb_batches_2 = nb_batches

n_epochs_2 = 20
LR_2 = LR

#Pretraining 3 constants

len_train_set_3 = 1000
len_test_set_3 = 100
                            
frame_size_3 = 60
n_digits_3 = 1
          
level_3 = level

batch_size_3 = batch_size
nb_batches_3 = nb_batches

n_epochs_3 = 40

#Pretraining 4 constants

len_train_set_4 = 1000
len_test_set_4 = 100
                            
frame_size_4 = 100
n_digits_4 = 2
          
level_4 = level

batch_size_4 = batch_size
nb_batches_4 = nb_batches

n_epochs_4 = 14

#Training 5 constants

len_train_set_5 = 1000
len_test_set_5 = 100
                            
frame_size_5 = 280
n_digits_5 = 5
          
level_5 = level

batch_size_5 = batch_size
nb_batches_5 = nb_batches

n_epochs_5 = 20




#To get the variables from the command
read_args(globals())

#Defining the directories and necessary variables
if folder != False :
	
		results_directory = folder+"/results/"
		models_directory = folder+"/models/"
		trained_models_directory = folder+"/trained_models/"

		checkpoint_path = folder+"/checkpoints/"
else:

	shadow_name = "shadow" if shadow else "no_shadow"
	photo_name = "_photo"*photo
	results_directory = shadow_name+photo_name+"/results/"

	models_directory = shadow_name+photo_name+"/models/"
	trained_models_directory = shadow_name+photo_name+"/trained_models/"

	checkpoint_path = shadow_name+photo_name+"/checkpoints/"

if not save:
    results_directory="trash/"
    models_directory="trash/"
    trained_models_directory="trash/"
    checkpoint_path = "trash/"
if all_epochs_same:
	n_epochs_0, n_epochs_1, n_epochs_2, n_epochs_3, n_epochs_4, n_epochs_5 = n_epochs,n_epochs,n_epochs,n_epochs,n_epochs, n_epochs








#Importing the custom loss
keras.losses.custom_loss = new_weighted_loss



#Importing and normalizing MNIST dataset
(digits_train, labels_train), (digits_test, labels_test) = keras.datasets.mnist.load_data()
digits_train = digits_train/255
digits_test = digits_test/255









#Pretraining 0 : 4x4 downsampled MNIST



if step_0 :




	#Generating the dataset

	train_data_0, train_labels_0 = generate_frames_dataset(digits_train, labels_train, 
				                       n_digits=n_digits_0, n_frames=len_train_set_0, 
				                       upsample=upsample_0, frame_size=frame_size_0, 
				                       downsample=downsample_0, 
				                       pool_size=pool_size_0, strides=strides_0, 
				                       final_size=final_size_0, 
				                       movie=movie_0, duration=movie_duration)

	test_data_0,test_labels_0 = generate_frames_dataset(digits_test, labels_test, 
		                               n_digits=n_digits_0, n_frames=len_test_set_0, 
		                               upsample=upsample_0, frame_size=frame_size_0, 
		                               downsample=downsample_0, pool_size=pool_size_0, strides=strides_0, 
		                               final_size=final_size_0, 
		                               movie=movie_0, duration=movie_duration)




	#Building and compiling the model

	unet_model_0 = build_photo_unet_model(final_size_0, movie_duration, photo, digit_max, depth, level_0, chan1_n)


	unet_model_0.summary()

	unet_model_0.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=["accuracy", my_accuracy, my_cat_accuracy])


	callback_0 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/0/",
		                                         save_weights_only=True,
		                                         monitor="accuracy",
		                                         mode="max",
		                                         save_best_only=True,
		                                         
		                                         verbose=1)


	#Training


	model_history_0 = unet_model_0.fit(train_data_0, train_labels_0 
		                         ,
		                        epochs=n_epochs_0,
		                        batch_size=batch_size,
		                        validation_data=[test_data_0, test_labels_0],
		                        callbacks=[callback_0])







	#Plotting the accuracies, the loss






	accuracy_0 = model_history_0.history["my_accuracy"]
	plt.figure()
	plt.title("step 0 accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.plot(accuracy_0)
	plt.savefig(results_directory+"accuracy_0")
	plt.close()

	cat_accuracy_0 = model_history_0.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 0 cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("cat_accuracy")
	plt.plot(cat_accuracy_0)
	plt.savefig(results_directory+"cat_accuracy_0")
	plt.close()




	loss_0 = model_history_0.history["loss"]
	plt.figure()
	plt.title("step 0 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_0)
	plt.savefig(results_directory+"loss_0")
	plt.close()






	#Plotting some example results of the predictions









	predictions_0 = unet_model_0.predict(test_data_0)

	k = 0
	test_image = test_data_0[k][:,:,4]

	fig0 = plt.figure()
	plt.title("the data")
	plt.imshow(test_image)
	plt.savefig(results_directory+"data_0")

	plt.close()

	fig1, axs1 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    axs1[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs1[digit//n_columns,digit%n_columns].imshow(test_labels_0[k][:,:,digit])
	plt.savefig(results_directory+"labels_0")

	plt.close()
	    
	fig2, axs2 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    prediction_0 = predictions_0[k][:,:,digit]
	    axs2[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs2[digit//n_columns,digit%n_columns].imshow(prediction_0)
	    
	plt.savefig(results_directory+"predictions_0")

	plt.close()


	#Saving the models to use the weights for the next training
	unet_model_0.save(trained_models_directory+"unet_0")








#Pretraining 1 : 8x8 downsampled MNIST


if step_1 :



	#Generating the dataset






	train_data_1, train_labels_1 = generate_frames_dataset(digits_train, labels_train, 
		                               n_digits=n_digits_1, n_frames=len_train_set_1, 
		                               upsample=upsample_1, frame_size=frame_size_1, 
		                               downsample=downsample_1, pool_size=pool_size_1, strides=strides_1, 
		                               final_size=final_size_1, 
		                               movie=movie_1, duration=movie_duration)

	test_data_1,test_labels_1 = generate_frames_dataset(digits_test, labels_test, 
		                               n_digits=n_digits_1, n_frames=len_test_set_1, 
		                               upsample=upsample_1, frame_size=frame_size_1, 
		                               downsample=downsample_1, pool_size=pool_size_1, strides=strides_1, 
		                               final_size=final_size_1, 
		                               movie=movie_1, duration=movie_duration)




	#Building and compiling the model




	unet_model_1 = build_photo_unet_model(final_size_1, movie_duration, photo, digit_max, depth, level_1, chan1_n)

	unet_model.summary()

	unet_model_1.load_weights(checkpoint_path+"/0/",
		                 ).expect_partial()

	unet_model_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR), 
		          loss=new_weighted_loss,
		          metrics=["accuracy", my_accuracy, my_cat_accuracy])

	callback_1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/1/",
		                                         save_weights_only=True,
		                                         monitor="accuracy",
		                                         mode="max",
		                                         save_best_only=True,
		                                       
		                                         verbose=1)





	#Training






	model_history_1 = unet_model_1.fit(train_data_1, train_labels_1,
						batch_size=batch_size,
				                epochs=n_epochs_1,
				                validation_data=[test_data_1, test_labels_1],
		                        	callbacks=[callback_1])





	#Plotting the accuracies and loss

	accuracy_1 = model_history_1.history["my_accuracy"]
	plt.figure()
	plt.title("step 1 accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.plot(accuracy_1)
	plt.savefig(results_directory+"accuracy_1")
	plt.close()

	cat_accuracy_1 = model_history_1.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 1 cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("cat_accuracy")
	plt.plot(cat_accuracy_1)
	plt.savefig(results_directory+"cat_accuracy_1")
	plt.close()



	loss_1 = model_history_1.history["loss"]
	plt.figure()
	plt.title("step 1 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_1)
	plt.savefig(results_directory+"loss_1")
	plt.close()





	#Plotting some examples of predictions







	predictions_1 = unet_model_1.predict(test_data_1)


	k = 0

	fig3 = plt.figure()
	plt.title("the data")
	plt.imshow(test_data_1[k][:,:,4])
	plt.savefig(results_directory+"data_1")

	plt.close()

	fig4, axs4 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    axs4[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs4[digit//n_columns,digit%n_columns].imshow(test_labels_1[k][:,:,digit])
	plt.savefig(results_directory+"labels_1")

	plt.close()

	fig5, axs5 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    prediction_1 = predictions_1[k][:,:,digit]
	    axs5[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs5[digit//n_columns,digit%n_columns].imshow(prediction_1)
	    
	plt.savefig(results_directory+"predictions_1")

	plt.close()



	unet_model_1.save(trained_models_directory+"unet_1")






#Pretraining 2 : MNIST




if step_2 :




	#Generating the dataset







	train_data_2, train_labels_2 = generate_frames_dataset(digits_train, labels_train, 
		                               n_digits=n_digits_2, n_frames=len_train_set_2, 
		                               upsample=upsample_2, frame_size=frame_size_2, 
		                               downsample=downsample_2, pool_size=pool_size_2, strides=strides_2, 
		                               final_size=final_size_2, 
		                               movie=movie_2, duration=movie_duration)

	test_data_2,test_labels_2 = generate_frames_dataset(digits_test, labels_test, 
		                               n_digits=n_digits_2, n_frames=len_test_set_2, 
		                               upsample=upsample_2, frame_size=frame_size_2, 
		                               downsample=downsample_2, pool_size=pool_size_2, strides=strides_2, 
		                               final_size=final_size_2, 
		                               movie=movie_2, duration=movie_duration)






	#Building and compiling the model






	unet_model_2 = build_photo_unet_model(final_size_2, movie_duration, photo, digit_max, depth, level_2, chan1_n)

	unet_model_2.summary()

	unet_model_2.load_weights(checkpoint_path+"/1/",
		                 ).expect_partial()

	unet_model_2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=["accuracy", my_accuracy, my_cat_accuracy])

	callback_2 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/2/",
		                                         save_weights_only=True,
		                                         monitor="accuracy",
		                                         mode="max",
		                                         save_best_only=True,
		                                         verbose=1)






	#Training






	model_history_2 = unet_model_2.fit(train_data_2, train_labels_2 #.astype(np.float32)
		                         ,
		                        epochs=n_epochs_2,
		                        batch_size=batch_size,
		                        validation_data=[test_data_2, test_labels_2],
		                        callbacks=[callback_2])




	#Plotting the accuracies and the loss




	accuracy_2 = model_history_2.history["my_accuracy"]
	plt.figure()
	plt.title("step 2 accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.plot(accuracy_2)
	plt.savefig(results_directory+"accuracy_2")
	plt.close()

	cat_accuracy_2 = model_history_2.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 2 cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("cat_accuracy")
	plt.plot(cat_accuracy_2)
	plt.savefig(results_directory+"cat_accuracy_2")
	plt.close()


	loss_2 = model_history_2.history["loss"]
	plt.figure()
	plt.title("step 2 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_2)
	plt.savefig(results_directory+"loss_2")
	plt.close()






	#Plotting some example predictions






	predictions_2 = unet_model_2.predict(test_data_2)


	k = 0

	fig6 = plt.figure()
	plt.title("the data")
	plt.imshow(test_data_2[k][:,:,4])
	plt.savefig(results_directory+"data_2")

	plt.close()



	fig7, axs7 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    axs7[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs7[digit//n_columns,digit%n_columns].imshow(test_labels_2[k][:,:,digit])
	plt.savefig(results_directory+"labels_2")

	plt.close()

	fig8, axs8 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    prediction_2 = predictions_2[k][:,:,digit]
	    axs8[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs8[digit//n_columns,digit%n_columns].imshow(prediction_2)
	plt.savefig(results_directory+"predictions_2")

	plt.close()


	unet_model_2.save(trained_models_directory+"unet_2")






#Pretraining 3 (1 digit on a 60x60 frame)



if step_3 :




	#Generating the dataset




	train_data_3, train_labels_3 = generate_movie_dataset_5(digits_train, labels_train, 
		                    n_movies=len_train_set_3, duration=movie_duration,
		                    frame_size=frame_size_3, n_digits=n_digits_3,
		                    depth = depth, level = level_3,
		                    shadow = shadow, shadow_ratio = shadow_ratio, light_intensity = light_intensity,
		                    max_jump=max_jump,speed=speed,
		                    digit_max = digit_max)

	test_data_3,test_labels_3 = generate_movie_dataset_5(digits_test, labels_test, 
		                    n_movies=len_test_set_3, duration=movie_duration,
		                    frame_size=frame_size_3, n_digits=n_digits_3,
		                    depth = depth, level = level_3,
		                    shadow = shadow, shadow_ratio = shadow_ratio, light_intensity = light_intensity,
		                    max_jump=max_jump,  speed=speed,
		                    digit_max = digit_max
		                    )






	#Building and compiling the model





	unet_model_3 = build_photo_unet_model(frame_size_3, movie_duration, photo, digit_max, depth, level_3, chan1_n)

	unet_model_3.summary()

	unet_model_3.load_weights(checkpoint_path+"/2/",
		                 ).expect_partial()

	unet_model_3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=["accuracy", my_accuracy, my_cat_accuracy])

	callback_3 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/3/",
		                                         save_weights_only=True,
		                                         monitor="accuracy",
		                                         mode="max",
		                                         save_best_only=True,
		                                         
		                                         verbose=1)





	#Training




	model_history_3 = unet_model_3.fit(train_data_3, train_labels_3 #.astype(np.float32)
		                         ,
		                        epochs=n_epochs_3,
		                        batch_size=batch_size,
		                        validation_data=[test_data_3, test_labels_3],
		                        callbacks=[callback_3]
		                        )




	#Plotting the accuracies and the loss




	accuracy_3 = model_history_3.history["my_accuracy"]
	plt.figure()
	plt.title("step 3 accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.plot(accuracy_3)
	plt.savefig(results_directory+"accuracy_3")
	plt.close()

	cat_accuracy_3 = model_history_3.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 3 cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("cat_accuracy")
	plt.plot(cat_accuracy_3)
	plt.savefig(results_directory+"cat_accuracy_3")
	plt.close()


	loss_3 = model_history_3.history["loss"]
	plt.figure()
	plt.title("step 3 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_3)
	plt.savefig(results_directory+"loss_3")
	plt.close()




	#Plotting some example results of the predictions





	predictions_3 = unet_model_3.predict(test_data_3)


	k = 0

	fig9 = plt.figure()
	plt.title("the data")
	plt.imshow(test_data_3[k][:,:,4])
	plt.savefig(results_directory+"data_3")

	plt.close()


	fig10, axs10 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    axs10[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs10[digit//n_columns,digit%n_columns].imshow(test_labels_3[k][:,:,digit])
	plt.savefig(results_directory+"labels_3")

	plt.close()

	fig11, axs11 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    prediction_3 = predictions_3[k][:,:,digit]
	    axs11[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs11[digit//n_columns,digit%n_columns].imshow(prediction_3)
	    
	plt.savefig(results_directory+"predictions_3")    

	plt.close()


	unet_model_3.save(trained_models_directory+"unet_3")







#Pretraining 4 (2 digits on a 100x100 frame)





if step_4 :





	#Generating the dataset




	train_data_4, train_labels_4 = generate_movie_dataset_5(digits_train, labels_train, 
		                    n_movies=len_train_set_4, duration=movie_duration,
		                    frame_size=frame_size_4, n_digits=n_digits_4,
		                    depth = depth, level = level_4,
		                    shadow = shadow, shadow_ratio = shadow_ratio, light_intensity = light_intensity,
		                    max_jump=max_jump,speed=speed,
		                    digit_max =digit_max
		                    )

	test_data_4,test_labels_4 = generate_movie_dataset_5(digits_test, labels_test, 
		                    n_movies=len_test_set_4, duration=movie_duration,
		                    frame_size=frame_size_4, n_digits=n_digits_4,
		                    depth = depth, level = level_4,
		                    shadow = shadow, shadow_ratio = shadow_ratio, light_intensity = light_intensity,
		                    max_jump=max_jump,  speed=speed,
		                    digit_max = digit_max
		                    )



	#Building and compiling the model




	unet_model_4 = build_photo_unet_model(frame_size_4, movie_duration, photo, digit_max, depth, level_4, chan1_n)

	unet_model_4.summary()

	unet_model_4.load_weights(checkpoint_path+"/3/",
		                 ).expect_partial()

	unet_model_4.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=["accuracy", my_accuracy, my_cat_accuracy])

	callback_4 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/4/",
		                                         save_weights_only=True,
		                                         monitor="accuracy",
		                                         mode="max",
		                                         save_best_only=True,
		                                         
		                                         verbose=1)


	#Training 




	model_history_4 = unet_model_4.fit(train_data_4, train_labels_4,
		                        epochs=n_epochs_4,
		                        batch_size=batch_size,
		                        validation_data=[test_data_4, test_labels_4],
		                        callbacks=[callback_4])





	#Plotting the accuracies and loss







	accuracy_4 = model_history_4.history["my_accuracy"]
	plt.figure()
	plt.title("step 4 accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.plot(accuracy_4)
	plt.savefig(results_directory+"accuracy_4")
	plt.close()

	cat_accuracy_4 = model_history_4.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 4 cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("cat_accuracy")
	plt.plot(cat_accuracy_4)
	plt.savefig(results_directory+"cat_accuracy_4")
	plt.close()


	loss_4 = model_history_4.history["loss"]
	plt.figure()
	plt.title("step 4 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_4)
	plt.savefig(results_directory+"loss_4")
	plt.close()





	#Plotting some examples of predictions






	predictions_4 = unet_model_4.predict(test_data_4)

	k = 0

	fig12 = plt.figure()
	plt.title("the data")
	plt.imshow(test_data_4[k][:,:,4])
	plt.savefig(results_directory+"data_4")

	plt.close()

	fig13, axs13 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    axs13[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs13[digit//n_columns,digit%n_columns].imshow(test_labels_4[k][:,:,digit])
	plt.savefig(results_directory+"labels_4")

	plt.close()

	fig14, axs14 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    prediction_4 = predictions_4[k][:,:,digit]
	    axs14[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs14[digit//n_columns,digit%n_columns].imshow(prediction_4)
	plt.savefig(results_directory+"predictions_4")

	plt.close()


	unet_model_4.save(trained_models_directory+"unet_4")









#Training 5 (5 digits on a 280x280 frame)




if step_5 :




	#Generating the dataset




	train_data_5, train_labels_5 = generate_movie_dataset_5(digits_train, labels_train, 
		                    n_movies=len_train_set_5, duration=movie_duration,
		                    frame_size=frame_size_5, n_digits=n_digits_5,
		                    depth = depth, level = level_5,
		                    shadow = shadow, shadow_ratio = shadow_ratio, light_intensity = light_intensity,
		                    max_jump=max_jump, speed=speed,
		                    digit_max = digit_max
		                    )

	test_data_5,test_labels_5 = generate_movie_dataset_5(digits_test, labels_test, 
		                    n_movies=len_test_set_5, duration=movie_duration,
		                    frame_size=frame_size_5, n_digits=n_digits_5,
		                    depth = depth, level = level_5,
		                    shadow = shadow, shadow_ratio = shadow_ratio, light_intensity = light_intensity,
		                    max_jump=max_jump, speed=speed,
		                    digit_max = digit_max
		                    )







	#Building and compiling the model







	unet_model_5 = build_photo_unet_model(frame_size_5, movie_duration, photo, digit_max, depth, level_5, chan1_n)

	unet_model_5.summary()

	unet_model_5.load_weights(checkpoint_path+"/4/",
		                 ).expect_partial()

	unet_model_5.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
		          loss=new_weighted_loss,
		          metrics=["accuracy", my_accuracy, my_cat_accuracy])

	callback_5 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path+"/5/",
		                                         save_weights_only=True,
		                                         monitor="accuracy",
		                                         mode="max",
		                                         save_best_only=True,
		                                         
		                                         verbose=1)


	#Training





	model_history_5 = unet_model_5.fit(train_data_5, train_labels_5,
		                        epochs=n_epochs_5,
		                        batch_size=batch_size,
		                        validation_data=[test_data_5, test_labels_5],
		                        callbacks=[callback_5])





	#Plotting the accuracies and the loss






	accuracy_5 = model_history_5.history["my_accuracy"]
	plt.figure()
	plt.title("step 5 accuracy")
	plt.xlabel("epochs")
	plt.ylabel("accuracy")
	plt.plot(accuracy_5)
	plt.savefig(results_directory+"accuracy_5")
	plt.close()

	cat_accuracy_5 = model_history_5.history["my_cat_accuracy"]
	plt.figure()
	plt.title("step 5 cat_accuracy")
	plt.xlabel("epochs")
	plt.ylabel("cat_accuracy")
	plt.plot(cat_accuracy_5)
	plt.savefig(results_directory+"cat_accuracy_5")
	plt.close()


	loss_5 = model_history_5.history["loss"]
	plt.figure()
	plt.title("step 5 loss")
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.plot(loss_5)
	plt.savefig(results_directory+"loss_5")
	plt.close()




	#Plotting some example predictions





	predictions_5 = unet_model_5.predict(test_data_5)

	k = 0

	fig15 = plt.figure()
	plt.title("the data")
	plt.imshow(test_data_5[k][:,:,4])
	plt.savefig(results_directory+"data_5")

	plt.close()


	fig16, axs16 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    axs16[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs16[digit//n_columns,digit%n_columns].imshow(test_labels_5[k][:,:,digit])
	plt.savefig(results_directory+"labels_5")

	plt.close()

	fig17, axs17 = plt.subplots(n_lines, n_columns, figsize=(plot_size,plot_size))

	for digit in range(digit_max+1):
	    
	    prediction_5 = predictions_5[k][:,:,digit]
	    axs17[digit//n_columns,digit%n_columns].set_title(str(digit))
	    axs17[digit//n_columns,digit%n_columns].imshow(prediction_5)
	plt.savefig(results_directory+"predictions_5")

	plt.close()

	unet_model_5.save(trained_models_directory+"unet_5")

