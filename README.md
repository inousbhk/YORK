# YORK

The objective of this project is to use a photoreceptor model to improve the performance of a Unet on an object detection task with dynamic lighting conditions. Versions used : 
- Python 3.9.12
- Tensorflow 2.7.0
- Scikit-learn 1.1.1
- CUDA 10.1
- Keras 2.7.0


This repository contains the scripts functions.py, Unet_1.py and Unet_2.py. 


- functions.py contains all the necessary functions to generate different types of datasets (simple frames, movies, with different light conditions...). Here is an overview of the most important ones :
  - generate_frames_dataset. Used to generate the data for the task 2 : still MNIST digits (downsampled or not) with spotlights or shadows. Can also generate larger frames. Here are the parameters :
    - digits_set : set of MNIST digits to use
    - labels_set : set of MNIST corresponding labels
    - n_digits : number of digits to put on the frame (default to 5)
    - n_frames : number of frames (or movies) to generate (default to 1000)
    - upsample : whether to put the digit in a frame larger than 28x28, bool (default to False) 
    - frame_size : frame_size after the upsampling, before the downsampling (default to 280) 
    - downsample : whether to downsample the digit, bool (default to False)
    - pool_size : for the downsampling (default to 7) 
    - strides : for the downsampling (default to 7)
    - final_size : frame size after the downsampling (default to 4)
    - movie : whether to transform the frame into a movie with identical frames, bool (default to False)
    - duration : if movie (default to 5)
    - perturbations : whether to add special lighting conditions, bool (default to False)
    - perturbations_time : if perturbations, when do they happen (default to "random", but must be an int = starting frame index)
    - perturbations_duration : if perturbations, in number of frames (default to 2)
    - half : if perturbations, whether to add the light conditions on half the frame or all the frame, bool (default to False)
    - spot_proba : if perturbations, probability for it to be a spotlight (o if only shadows, 0.5 for both, 1 for only spotlights, default to 0.5)
    - spot_factor : if there are spotlights, multiplying factor (default to 10, usually 10**n)
    - shadow_factor : if there are shadow multiplying factor (default 0.5)
    
  - generate_movie_dataset_5. Used to generate the data for the task 1 : moving digits on a larger frame. Can add a circular shadow or spotlight. Here are the parameters :
    - digits_set : set of MNIST digits to use
    - labels_set : set of MNIST corresponding labels
    - n_movies : number of movies to generate (default to 1000)
    - duration : (default to 5)
    - frame_size : (default to 280)
    - n_digits : number of digits to put on the frame (default to 5)
    - depth : number of blocks on each side of the U-net (default to 2)
    - level : level on the upsampling side to use for the output (default to 1 = last level, but could be useful to train separately on each level (0 then 1), with appropriately downsampled labels. Instead, I pretraind the whole model on downsampled data)
    - shadow : whether to add a shadow, bool (default to False)
    - shadow_ratio : ratio of shadow/total surface, that defines the circle radius (default to 0.2, because it corresponds to a reasonable circle radius)
    - light_intensity : multiplying factor for the shadow, can be used to create a spotlight if > 1 (default to 0.1)
    - max_jump : defines the number of possible directions for a digit's movement (default to 1 : only the closest positions are available -> 6 directions)
    - speed : defines how far the digit can go in one time step ( x += x + speed * direction , default to 1)
    
  - read_args. Allows us to define the constant parameters of the Unet script in the Python command, with the syntax --variable_name=value=type (see later in the Read_Me)
  
  - save_movie. Allows us to save a movie in a GIF format, to check that the data is as we want it. Parameters :
    - movie : dimension (duration, frame_size, frame_size)
    - title : string, name of the saved movie
    - directory : string, where to save the movie
    - time_interval : time between two frames in the movie. 50 is fast, 150 is slow (default to 70)
    
  - new_weighted_loss. Custom loss, that balances background pixels and digits pixels and computes the cross entropy loss (- mean of true_labels * weights * log(predictions))
  
  - my_accuracy. Custom accuracy, "non categorical". Computes the mean of exp(-abs(relative_error)), only taking into account the digit pixels and not the background predictions (because with the softmax activation, it is not necessary). This one could be improved by adding a multiplying factor inside the exponential, to get a better range (right now, between ~0.9 and 1)
  
  - my_cat_accuracy. Custom acuracy, "categorical". Computes the frequency of good predictions, choosing the maximum probability over the classes, only taking into account digit pixels, and not the background predictions.
  
  - build_photo_unet_model. Builds the model, with a Unet of depth 2, and an optional photoreceptor model in front of it. Parameters :
    - size : frame size of the input
    - duration : duration of the movies
    - photo : whether to add the PR model in front of the Unet or not, bool

  - class photoreceptor_DA_multichan_randinit. If you need more information, ask Saad, but this one is useful to toptimize the model : you can change the range of the parameters if some of them seem to not be adequate. To know it the range is adequate, you can plot the parameters during the training, see later.





- Unet_1 corresponds to the initial task (task 1), with 280x280 movies, moving digits, with 5 pretrainings before the real training. The dataset is generated with generate_movie_dataset_5. When called with a python command of the usual format (python Unet_1.py) it trains a Unet on task_2. A lot of parameters can be passed in the command. Here is a list with some explanation of the important parameters : 






- Unet_2 corresponds to the new task (task 2), with 28x28 movies, still digits, with 2 pretrainings before the real training. The dataset is generated with generate_frames_dataset.

When called with a python command of the usual format (python Unet_2.py) it trains a Unet on task_2 and stores the results in the corresponding folder. A lot of parameters can be passed in the command. Here is a list with some explanation of the important parameters : 

- Model parameters : 
	- photoreceptor model or not : --photo=True=bool if you want to run with the photoreceptor model in front of the Unet (default to False). If True, you can precise the number of channels --nchan_1=k=int (default to 9).

- Data parameters :
	- special light conditions or not : perturbations=True=bool (default to false). 
	- when the light conditions happen : perturbations_time=2=int, corresponding to the 3rd frame (default to "random")
	- whether it is a spotlight or a shadow : spot_proba=1=int (for only spotlight, can be 0.5=float for random spotlights and shadow, which is the default value, or 0=int for only shadows)
	- the light multiplying factor for the spotlight or the shadow, --spot_factor=10=int (default value) and shadow_factor=0.5=float (default value)

- Training parameters : 
  - number of epochs : --all_epochs_same=False=bool if you want all the training steps to have different numbers of epochs 9default to True). In this case, you must enter --n_epochs_0=m=int --n_epochs_1=n=int --n_epochs_2=k=int. Or you can let the default value for all_epochs_same (True) and just enter --n_epochs=n=int (default 1, to run a test).
  - learning rate : --LR=0.01=float (default to 0.001, best results)
  - batch size : --batch_size=64=int (default value)
  - number of batches : --nb_batches=1=int (default value)
		

- Results parameters 
	- whether to save the results or not : --save=True=bool (default value). If False, the necessary data (trained models' weights) will be saved in a folder called "trash" (that has to be created before hand). Sometimes when I am just running a test I put the results in trash, so that I dont overwrite important results.
	- folder to save the results : --folder=task_2/folder_name=str, default to False (folder name built with the parameters)
	- photoreceptor model's parameters to plot : --plot_params=[alpha,beta,gamma]=str. The parameters will be plot during the training on the same figure. Useful to change their range when tuning the PR model.
		
Those are the most used parameters, but every constant defined in the beginning of Unet_2 can be changer using the same syntax --parameter_value=value=type. (I am passing the type because I had some problems with the read_args function automatically finding the type so I just pass it). In the Unet script, the values of the parameters are changed when the function read_args(globals) is called. After that, the directories are defined depending on the parameters, and the given folder name.
