import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random as random
import matplotlib.animation as animation
import pandas as pd
from random import sample
from tensorflow.keras import Sequential
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.layers import Input, InputLayer

from tensorflow.keras import backend as K
from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, Dense, Activation, Flatten, Reshape, MaxPool3D, MaxPool2D, Permute, BatchNormalization, GaussianNoise,DepthwiseConv2D, Dropout, LayerNormalization
from tensorflow.keras.regularizers import l1, l2
import math
import sys



digit_max=10






#Dataset generating functions (Ines)









def generate_frames_dataset(digits_set, labels_set, n_digits=5, n_frames=1000, 
                            upsample=True, frame_size=280, 
                            downsample=False, pool_size=7, strides=7, final_size=4, 
                            movie=False, duration=5, 
                            perturbations = False, perturbations_time="random", perturbations_duration=2, half=False,
                            spot_proba = 1, spot_factor=100, shadow_factor=0.5,
                           digit_max = 10):
    
    """Generates a dataset of frames with n_digits, with variable frame_size, and possibilitiy to downsample, or movies (with identical frames) if movie=True. Adds light perturbations if perturbations=True, that can be 2 frames long homogenous or half frame light or shadow, at a specific time or random time. 
    
    Args:
    digits_set: np.array of MNIST digits
    labels_set: np.array of int type corresponding to the labels
    n_digits: int
    n_frames: int
    upsample; bool, True or False
    frame_size: if upsample, size of the frame after the upsampling, before the downsampling
    downsample: bool, True or False
    pool_size: int, if downsample
    strides: int, if downsample
    final_size: int
    movie: bool, True or False
    duration: int, if movie
    perturbations: bool, whether to add spotlights and shadows
    perturbations_time: str or int, when to start the 2-frames long perturbations
    half: bool, whether to add the perturbations on only half of the frame (if False : homogeneous perturbation)
    spot_proba: float, between 0 and 1, probability that the perturbation is a spotlight (0.5 if you want equiprobable spotlights and shadows)
    spot_factor: float, by how much to multiply the light intensity
    shadow_factor: float, but how much to multiply the light intensity as well ( < 1 )
    digit_max: maximum of the classes labels, by default 10. Can be less, if you only want to work with some of the classes, and reduce the MNIST dataset you use.
   
    Returns:
    dataset: np.array of frames (n_frames, final_size, final_size) or movies (n_frames, final_size, final_size, duration)
    labels_set: np.array of set of digit_max frames, corresponding to the labels for each class (n_frames, final_size, final_size, digit_max+1)
    
    """
    
    
    digit_shape = np.shape(digits_set[0])
    
    digit_max = len(list(set(labels_set)))
    frames = []
    labels = []
    	
    	
    if upsample:
        
        for i in range(n_frames):
        
            frame = np.zeros((frame_size, frame_size), dtype="float32")
            label = [np.zeros((frame_size, frame_size), dtype="float32" ) for i in range(digit_max)]  #build empty frames and labels


            indexes = random.sample(range(len(digits_set)), n_digits)  #select the MNIST digits, and get the corresponding labels
            digits = digits_set[indexes]
            digits_labels = labels_set[indexes]

            positions = [random.sample(range(frame_size-digit_shape[0]-1),2) for i in range(n_digits)]  #choose the position of the digits
            background = np.ones((frame_size, frame_size), dtype="float32")

            if downsample :
                
                new_frame_size, new_digits, new_positions = little_downsample(frame_size, digits, positions, pool_size, strides)
                #computing the downsampled positions and digits
                frame, label = create_frame_and_labels(new_frame_size, new_digits, digits_labels, new_positions, digit_max)
            
            else:
                
                frame, label = create_frame_and_labels(frame_size, digits, digits_labels, positions, digit_max)
            
            if movie:
            
            	frame = [frame for i in range(duration)]  #making a movie out of identical frames
            	
            	if perturbations:  #adding spotlights and shadows
            	
            		frame = add_perturbations(frame, half, spot_proba, spot_factor, shadow_factor, perturbations_time, movie_duration, final_size)

            labels.append(label)
            frames.append(frame)
        
    else:  #if not upsampled
    
        for i in range(n_frames):

            n = random.randint(0,len(digits_set)-1)

            digit_label = labels_set[n]
            
            frame = digits_set[n]
            label = label_mnist(frame, digit_label, digit_max)

            
            if downsample:
                
                new_frame_size, new_digits, new_positions = little_downsample(frame_size, [frame], [[0,0]], pool_size, strides)
                frame, label = create_frame_and_labels(new_frame_size, new_digits, [digit_label], new_positions, digit_max)

            
            if movie:
            
            	frame = [frame for i in range(duration)]

            	if perturbations:
            	
            		frame = add_perturbations(frame, half, spot_proba, spot_factor, shadow_factor, perturbations_time, perturbations_duration, duration)
				
            labels.append(label)
            frames.append(frame)
            
    dataset = np.array(frames)
    labels_set = np.array(labels)

    #getting the dimensions right for the model
    if not movie:
        dataset = dataset[:,:,:,None]
        labels_set = np.transpose(labels_set, (0,2,3,1))
    

    dataset = np.transpose(dataset, (0,2,3,1))
    labels_set = np.transpose(labels_set, (0,2,3,1))
    
    return dataset, labels_set
 
 
 
 
 
 
 
 
 
 
 


def generate_movie_dataset_5(digits_set, labels_set, 
                            n_movies=1000, duration=5,
                            frame_size = 280, n_digits=5,
                            depth=2, level=1,
                            shadow = False, shadow_ratio = 0.2, light_intensity = 0.1,
                            max_jump=1, speed=1,
                            digit_max = 10
                            ):
    
    """Generates a dataset of movies with moving digits
    
    Args:
    digits_set: np.array of MNIST digits
    labels_set: np.array of the corresponding labels
    n_movies: int, number of movies
    duration: int, duration of the movie
    frame_size: int, size of the frame
    n_digits: int, numer of digits
    shadow: bool, True or False
    shadow_ratio: float, usually 1/10**n, shadow surface/total surface, defines the radius. Reasonable circle size with ratio=0.3
    light_intensity: float, usually 10**n, factor applied to pixels in the shadow
    
    Returns:
    movies: np.array of movies, shape (n_movies, frame_size, frame_size, duration)
    labels: np. array of frames labelling the last frame of each movie for each class
    
    """
    
    radius = int(frame_size*np.sqrt(shadow_ratio/np.pi))
    
    movies, labels = [], []
    
    label_size = frame_size/2**(depth-level-1)

    digit_size = len(digits_set[0][0])
    
    for n in range(n_movies):
        
        # select the digits
        
        indexes = random.sample(range(len(digits_set)), n_digits)
        
        digits = digits_set[indexes]
        digits_labels = labels_set[indexes]
        
        #Initial values for the positions and directions of the digits
		
        positions = [random.sample(range(speed*max_jump, frame_size-digit_size-speed*max_jump),2) for i in range(n_digits)]
        directions =  [non_null_sample(max_jump, 2) for i in range(n_digits)]

        #create the movie and the movie labels
        
        movie = []
        
        for time in range(duration):

            for k, digit in enumerate(digits):
                
                #Compute the new position
                
                position = new_position(positions[k], directions[k], speed)
                positions[k] = position
                
                #Compute the next position to check it's going to be fine
                
                next_position = new_position(positions[k], directions[k], speed)

                while out_of_frame(next_position, frame_size, digit_size):
                
                    #Correct the direction if next position is out of frame
                    
                    rebound_point = closest_frontier(next_position, frame_size, digit_size)
                    directions[k] = mirror_direction(directions[k], rebound_point, frame_size)
                    next_position = new_position(positions[k], directions[k], speed)

                if in_the_center(positions[k], frame_size, speed, digit_size, max_jump):
                
                    #Randomly change the direction, if far enough from the walls
                    directions[k] = change_direction(directions[k], max_jump)
                    
            #Create the frame with the positions of the digits       
            frame = create_frame(frame_size, digits, positions)
            

            if shadow:
                frame = add_shadow(frame, radius, light_intensity)
                
            movie.append(frame)
            
            if time == duration-1 :
            
                    #only label the last time step
            
                    movie_labels = create_labels(frame_size, digits, digits_labels, positions, digit_max)

        movies.append(movie)
        labels.append(movie_labels)
        
    #Get the right shape for the model
    movies = np.transpose(movies, (0,2,3,1))
    labels = np.transpose(labels, (0,2,3,1))
        
    return np.array(movies), np.array(labels)









#Auxiliary functions for the generating functions (Ines)















def add_perturbations(frame, half, spot_proba, spot_factor, shadow_factor, perturbations_time, perturbations_duration, movie_duration):

	"""
	
	Adds light level variations to a frame movie. Either on all the frame, or only half, either spotlight or shadow or both with a probability.
	
	Args:
	frame: 28x28 np array
	half: boolean
	spot_proba: 0 if only shadows, 1 if only spotlights, 0.5 if randomly one or the other.
	spot_factor: factor to multiply the pixel values if spotlight, usually 10**n
	shadow_factor: factor to multiply pixel values if shadow, usually 1/10**n
	perturbations_time: "random" by default, or int, has to be <= movie_duration-perturbations_duration
	perturbations_duration: int, in number of frames
	movie_duration: int
	final_size: int, size of the frame
	
	Returns:
	frame: modified frame
	
	"""
	frame_size = len(frame)

	if perturbations_time=="random":
	
		start_time = random.randint(0,movie_duration-perturbations_duration)
		
	else:
		start_time = perturbations_time
		
	end_time = start_time + perturbations_duration -1
	
	random_nb = np.random.random() #to decide if spotlight or shadow


	if half:
		random_half = np.random.randint(0,2)

		if random_half==1: #low half perturbated
		
			new_frames = []
			for k in range(perturbations_duration):
			
				new_frame = np.vstack([frame[start_time+k][:int(frame_size/2)] * ( (random_nb < spot_proba)*spot_factor + (random_nb >= spot_proba)*shadow_factor ) , frame[start_time+k][int(frame_size/2):]])
				new_frames.append(new_frame)
				
		else: #top half perturbated
		
			new_frames = []
			for k in range(perturbations_duration):
			
				new_frame = np.vstack([frame[start_time+k][:int(frame_size/2)], frame[start_time+k][int(frame_size/2):] * ( (random_nb < spot_proba)*spot_factor + (random_nb >= spot_proba)*shadow_factor )])
				new_frames.append(new_frame)

    			
	else:  #homogeneous perturbation
	
		new_frames = []
		for k in range(perturbations_duration):
		
			new_frame = frame[start_time+k] * ( (random_nb < spot_proba)*spot_factor + (random_nb >= spot_proba)*shadow_factor )
			new_frames.append(new_frame)

	for k in range(perturbations_duration):
	
		frame[start_time+k] = new_frames[k]

	return frame













def add_shadow(frame, radius, intensity):
    
    """Adds a shadow on a frame.
    
    Args:
    frame: np.array
    radius: float
    intensity: factor applied to the pixels in the shadow
    
    Returns:
    shad_frame: np.array of the frame with the shadow
    
    """
    
    frame_size = len(frame)
    shad_frame = frame.copy()
    
    for i,line in enumerate(frame):
        for j,pixel in enumerate(line):
            
            x,y  = i - frame_size/2, j - frame_size/2
            
            if np.sqrt(x**2 + y**2) <= radius :
                shad_frame[i,j] = intensity*frame[i,j]
                
    return shad_frame
















def create_labels(frame_size, digits, digits_labels, positions, digit_max):


    """Create the labels corresponding to a frame.
    
    Args:
    frame_size: float
    digits: MNIST digits present on the frame
    digits_labels: labels of the digits
    positions: 2-uple (x,y) for each digit
    digit_max: 10
    
    Returns:
    labels:frames corresponding to each class
    
    """
    
    labels = np.array([np.zeros((frame_size, frame_size)) for i in range(digit_max)]+ [np.ones((frame_size, frame_size))])

    for k, digit in enumerate(digits):

        
        frame = labels[digits_labels[k]]



        label_k = insert(digit, frame, positions[k])


        labels[digits_labels[k]] = label_k


        labels[-1] = np.maximum(labels[-1]-labels[digits_labels[k]], 0)
        
    return labels

















def label_mnist(digit_frame, digit, digit_max=10):

	
	"""
	
	Creates the labels frames corresponding to an MNIST digit
	
	Args:
	digit_frame: 28x28 np array, MNIST digit directly
	digit: int, MNIST label
	digit_max: int, default to 10
	
	Returns:
	labels: digit_max+1 frames, shape (digit_max+1, 28,28), with zeros on every frame but the digit frame, where the non zero MNIST pixels are =1.
	
	"""
	
	labels = [np.zeros(np.shape(digit_frame), dtype="float32") for i in range(digit_max+1)]

	label = np.zeros(np.shape(digit_frame), dtype="float32")
	for i in range(len(digit_frame)):
		for j in range(len(digit_frame[i])):
	    		if digit_frame[i][j] != 0:
	    			label[i][j] = 1

	labels[digit] = label
	labels[-1] = 1-label

	return labels















def insert(digit, frame, position):
    
    """
    
    Adds a digit in the frame. If there is already a digit, it has priority (the other is "hidden" behind)
    
    Args:
    digit: 28x28 array
    frame: array
    position: x,y for the digit's top left corner
    
    Outputs:
    frame: array
    
    """

    max_value = np.max(digit)
    for i, line in enumerate(digit):

        for j, pixel in enumerate(line):

            frame_value = frame[position[0]+i, position[1]+j]
            
            if frame_value == 0 :
                frame[position[0]+i, position[1]+j] = pixel
                
            #else : already a prioritary digit here

    
    return frame















def create_frame_and_labels(frame_size, digits, digits_labels, positions, digit_max):
    
    """Create a frame and the corresponding labels.
    
    Args:
    frame_size: int
    digits: np.array of MNIST digits to put on the frame
    digits_labels: np.array of int labels corresponding to the digits
    positions: 2-uple (x,y) for each digit
    digit_max: 10
    
    Returns:
    frame: the created frame
    labels: the associated label frames for each class
    
    """
    
    frame = np.zeros((frame_size, frame_size))
    background = np.ones((frame_size, frame_size))
    labels = [np.zeros((frame_size, frame_size)) for i in range(digit_max)]
    
    for k, digit in enumerate(digits):
        
        digit_label = digits_labels[k]
        position = positions[k]
        
        frame, labels[int(digit_label)], background = insert_and_label(digit, digit_label, frame, labels[int(digit_label)], position, background)
        
    labels.append(background)
        
    return np.array(frame), np.array(labels)

















def insert_and_label(digit, digit_label, frame, proba, position, background):

    """
    
    Adds a digit to a frame and to the corresponding labels
    
    Args:
    digit: MNIST digit
    digit_label: int
    frame: frame to insert the digit in
    proba: label drame corresponding to the digit
    position: (x,y) digit position
    background: background frame in the labels
    
    Returns:
    frame: modified frame
    new_proba: modified label frame
    background: modified background frame
    
    """
    

    new_proba = proba.copy()
    max_value = np.max(digit)
    
    for i, line in enumerate(digit):
        for j, pixel in enumerate(line):
            
            frame_value = frame[position[0]+i, position[1]+j]
            
            if frame_value == 0:
                frame[position[0]+i, position[1]+j] = pixel
                
            #else : already a digit there, priority to the first arrived
                
            if pixel != 0:
                
                new_proba[position[0]+i][position[1]+j] = 1
                background[position[0]+i][position[1]+j] = 0
                #where a non-zero digit pixel is inserted, a one is inserted in the same position of the corresponding matrix


    
    return frame, new_proba, background

















def downsample_digit(digits_list, 
                     pool_size, strides
                    ):
                    
    """
    
    Downsamples digits
    
    Args:
    digits_list: list of MNIST digits
    pool_size: int, for the downsampling
    strides: int, for the downsampling
    
    Returns:
    down_digits: list of downsampled digits
    new_digit_size: int
    
    """
    
    down_digits = []
    
    digit_size = len(digits_list[0][0])

    
    for digit in digits_list :


        down_digit = []
        line = 0

        while line < len(digit[0])-pool_size+1 :
            new_line = []
            column = 0
            while column < len(digit[0])-pool_size+1 :
        
                new_line.append(np.max( digit[line:line+pool_size, column:column+pool_size]))
                column += strides
        
            line += strides
            down_digit.append(new_line)
        
        down_digit= np.array(down_digit)


        down_digits.append(down_digit)
        new_digit_size = len(down_digits[0])
        
    return down_digits, new_digit_size












def little_downsample(frame_size, digits, positions, pool_size, strides):
    
    """Computes the new frame size, the new MNIST digits and their new positions after the downsampling. 
    Allows to create the downsampled frame rather then downsampling a frame, faster.
    
    Args:
    frame_size; int, before downsampling
    digits: MNIST digits
    positions: 2-uple (x,y) for each digit
    pool_size: int
    strides: int
    
    Returns:
    down_frame_size: int, new frame size
    down_digits: np.array of new digits
    down_positions: 2-uple of new (x,y) position for each digit
    
    """
    
    digit_size = len(digits[0][0])
    
    down_frame_size = (frame_size-pool_size)/strides +1
    down_positions = []
    down_digits = []
    
    for position in positions:
        x,y = position
        down_x, down_y = x//strides, y//strides
        down_positions.append([down_x,down_y])
        
        
    down_digits, digit_size = downsample_digit(digits, pool_size, strides)

        
    return int(down_frame_size), down_digits, down_positions
















def create_frame(frame_size, digits, positions):
    
    """
    
    Creates the frame zith all the digits.
    
    Args:
    frame_size: for a square frame
    digits: list of 28x28 arrays
    positions: list of x,y corresponding to the digits' top left corners, in the same order
    
    Outputs:
    frame: 280x280 array
    
    """
    
    frame = np.zeros((frame_size, frame_size))
    for k, digit in enumerate(digits):
        position = positions[k]
        frame = insert(digit, frame, position)
    return frame

















def mirror_direction(direction, position, frame_size):
    
    """
    
    Finds the new direction when a digit rebounds on a wall.
    
    Args:
    direction: current direction of the digit [dx,dy] 
    position: current position of the digit [x,y]
    frame_size: for a square frame
    
    Outputs:
    direction: [dx,dy]
    
    """
    
    dx, dy = direction
    x,y = position
    if (x==0 or x==frame_size-1) and y!=0 and y!=frame_size-1:
        dx = -dx
    if (y==0 or y==frame_size-1) and x!=0 and x!=frame_size-1:
        dy = -dy
    if (x==0 or x==frame_size-1) and (y==0 or y==frame_size-1):
        dx, dy = -dx, -dy
    return [dx, dy]


















def change_direction(direction, max_jump):
    
    """
    
    Generates a slight change in direction, called with a small probability.
    
    Args:
    direction: current direction [dx,dy]
    max_jump: initially defined constant
    
    Outputs:
    direction: new direction[dx,dy]
    
    """
    
    random_nb = random.random()
    if random_nb < 0.05:
        direction[0] = min(max_jump, max(-max_jump,direction[0]+1))
        # to check that still between -max_jump and max_jump
    elif random_nb >= 0.05 and random_nb < 0.1 :
        direction[0] = min(max_jump, max(-max_jump,direction[0]-1))
    if random_nb >= 0.1 and random_nb < 0.15:
        direction[1] = min(max_jump, max(-max_jump,direction[1]+1))
    elif random_nb >= 0.15 and random_nb < 0.2 :
        direction[1] = min(max_jump, max(-max_jump,direction[1]-1))
    return direction



















def closest_frontier(position, frame_size, digit_size):

	"""
	
	Finds the closest point on the walls to a digit
	
	Args:
	position: (x,y) position of the top left corner of the digit
	frame_size: int
	digit_size: int
	
	Returns:
	(x,y) position of the closest point on a wall
	
	"""
    
	x,y = position
	neighbor_frontiers = [(frame_size-1,y), (0,y), (x, frame_size-1), (x,0)]
	distances = []
	distances.append(frame_size - (x+digit_size))
	distances.append(x)
	distances.append(frame_size - (y+digit_size))
	distances.append(y)
	return neighbor_frontiers[distances.index(min(distances))]



















def non_null_sample(max_value, number):

	"""
	
	Selects a number-uple of values different from (0,...,0) for the directions, usually (dx,dy) because 2D
	
	Args:
	max_value: int, max value to choose
	number: number of numbers to pick
	
	Returns:
	number-uple of values not all zero, usually (dx,dy)
	
	"""
    
	values = list(np.arange(-max_value, max_value+1))
	sample = []
	for i in range(number-1):
		sample.append(random.choice(values))
	#for the last value, we cannot add a zero if there are only zero in the selected values
	if list(set(sample)) != [0]:
		sample.append(random.choice(values))
	else:
		values.remove(0)
		sample.append(random.choice(values))
	return sample

















def new_position(position, direction, speed):

	"""
	
	Computes the new position from current position, direction and speed
	
	Args:
	position: (x,y) int 
	direction: (dx, dy) int
	speed: int
	
	Returns:
	(x,y) new position : x += speed*dx
	
	"""
	
	x,y = position
	dx, dy = direction
	return [x+speed*dx, y+speed*dy]

















def out_of_frame(position, frame_size, digit_size):

	"""
	
	Checks whether a position would go out of the frame
	
	Args:
	position: (x,y) int
	frame_size: int
	digit_size: int
	
	Returns:
	bool: True if out of frame
	
	"""
	for i, dimension in enumerate(position) :

		if dimension + digit_size  > frame_size:
			return True
		if dimension < 0:
			return True
	return False


















def in_the_center(position, frame_size, speed, digit_size, max_jump):
	"""
	
	Checks whether a digit is far enough from the walls (to randomly change direction without danger)
	
	Args:
	position: (x,y) int
	frame_size: int
	speed: int
	digit_size: int
	max_jump: int
	
	Returns:
	bool: True if far enough from the walls
	
	"""
	
	security_distance = abs(speed*max_jump)+ digit_size
	for dimension in position :
		if abs(frame_size-dimension)<security_distance or dimension < security_distance:
	    		return False
	return True


















def save_movie(movie, title, directory, time_interval=70):
    
    
    """
    Saves a movie in a directory in GIF format
    
    Args:
    movie: np.array of dimension (n_frames, frame_size, frame_size)
    title: string
    directory: str of shape "folder1/folder2/"
    time_interval: int, between two frames. 50 is fast, 150 is slow
    
    Returns:
    nothing, the movie is saved.
    
    """
    
    fig, ax  = plt.subplots()
    ims = []
    for i, image in enumerate(movie):
        im = ax.imshow((image), animated=True)
        if i == 0:
            ax.imshow(image) 
        ims.append([im])

    gif = animation.ArtistAnimation(fig, ims, interval=time_interval, blit=True,
                                repeat_delay=1000)
    gif.save(directory+title+".gif")
    
    
    
    
    
    
    
    
    
#Custom metrics (Ines)














@tf.function(experimental_relax_shapes=True)
def new_weighted_loss_0(y_true, y_pred):

    """
    Counts the number of zeros and ones pixels on the labels, then applies the weights on all the frames based on the background to balance the digits and background classes (because a lot more zeros). Then computes the weighted cross-entropy loss (- mean of true_labels * weights * log(predictions) )
    
    Args:
    y_true: true labels
    y_pred: predictions
    
    Returns:
    Cross-entropy loss
    
    """

    background = y_true[:, :,-1]
    w1 = tf.reduce_sum(background)/tf.cast(tf.size(background), tf.float32)
    w2 = 1-w1
    w = background*w2 + (1-background)*w1
    
    #loss = tf.reduce_mean(w[:, :, None]*tf.square(tf.subtract(y_true, y_pred))) 
    #to broadcast (over channels)
    #tf.print(tf.math.log(y_pred))

    y_pred = tf.clip_by_value(y_pred, 0.0001, 0.9999)
    term = w[:,:, None]* (y_true)* tf.math.log(y_pred) 
    loss =tf.reduce_mean(term)
    #tf.print(loss)

    return tf.math.abs(loss)
    
    
    
    
    
    
    
    
    
    
    
    
    
@tf.function
def new_weighted_loss(y_true, y_pred):

    """
    Counts the number of zeros and ones pixels on the labels, then applies the weights on each frame separately to balance the zeros and ones classes (because a lot more zeros). Then computes the weighted cross-entropy loss (- mean of true_labels * weights * log(predictions) )
    
    Args:
    y_true: true labels
    y_pred: predictions
    
    Returns:
    Cross-entropy loss
    
    """

    w1 = tf.reduce_sum(y_true)/tf.cast(tf.size(y_true), tf.float32)  #proportion of ones pixels
    w2 = 1-w1   #proportion of zeros pixels
    
    N = len(y_true[0,0,0])  # number of classes
    weights = tf.TensorArray(tf.float32, size=N)
    
    #now we build the weights matrix
    
    for i in range(N):
    
    	frame = y_true[:,:,:,i]
    	weights_frame = frame*w2 + (1-frame)*w1   #ones pixels get the zeros pixels proportion as their weight, for each frame
    	
    	weights = weights.write(i, weights_frame)
    	
    weights = tf.transpose(weights.stack(), (1, 2, 3, 0))
        
    y_pred = tf.clip_by_value(y_pred, 0.0001, 0.9999)
    #to avoid log(0) that was causing an error
    term = weights * (y_true)* tf.math.log(y_pred) 
    loss = - tf.reduce_mean(term)

    return loss















@tf.function(experimental_relax_shapes=True)
def my_accuracy(y_true, y_pred):

	"""
	Computes the mean of exp(-abs(relative_error)), only taking into account digit pixels, and not the background. Idea: to get a better range of the accuracy values, we could add a multiplying factor inside the exp.
	
	Args:
	y_true: true labels
	y_pred: predictions
	
	Returns:
	non-categorical accuracy
	
	"""
	
	y_true_digits = y_true[:,:,:,:-1]
	y_pred_digits = y_pred[:,:,:,:-1]
	background = y_true[:,:,:,:-1]
	n_classes = y_true.get_shape()[-1]-1
	
	#What I used in my runs
	similarity = tf.math.exp(- tf.math.abs( y_pred_digits - y_true_digits) )
	
	#What I realized on the presentation day would have been better
	#1. Using the relative error
	#2. Using a factor to decide how low the accuracy can go (right now, betwen 0.9 and 1...)
	#similarity = tf.math.exp(- tf.math.abs( factor * (y_pred_digits - y_true_digits)/y_true_digits )
	


	digit_index = tf.math.argmax(tf.math.reduce_max(y_true[:,:,:,:-1], axis=(0,1,2))) 
	digit_frame = y_true_digits[:,:,:,digit_index]	
	
	weighted_similarity = digit_frame[:,:,:,None] * similarity



	my_accuracy = tf.cast(tf.reduce_sum( weighted_similarity ), tf.float32) / tf.cast(tf.math.count_nonzero(digit_frame) * n_classes, tf.float32)
	
	return my_accuracy

	
	
	
	
	
	
	
	
	
	
	
	
@tf.function(experimental_relax_shapes=True)
def my_cat_accuracy(y_true, y_pred):

	"""
	Computes the frequency of good predictions, choosing the maximum probability over the classes, only taking into account digit pixels, and not the background.
	
	Args:
	y_true: true labels
	y_pred: predictions
	
	Returns:
	categorical accuracy
	
	"""

	background = y_true[:,:,:,-1]
	
	#we "force the network to choose" by taking the maximum over the classes
	indexes_true = tf.math.argmax(y_true,axis=3)
	indexes_pred = tf.math.argmax(y_pred,axis=3)
	
	#the "indexes" are just one frame wih values between 0 and 10, indicating the class with the maximum probability for each pixel

	#find where the digit actually is
	digit_index = tf.math.argmax(tf.math.reduce_max(y_true[:,:,:,:-1], axis=(0,1,2))) 
	
	#the reduce_max is list of n_classes element, excluding the background. the max is the frame with the actual digit (the others are only zeros)

	#we count the pixels where the predicted class is correct, and is the digit (not background) 
	acc = tf.math.reduce_sum(tf.cast((indexes_true == indexes_pred) & (indexes_true==digit_index), tf.float32)) / tf.cast(tf.math.count_nonzero(1-background), tf.float32)

	return acc









#Code for the photoreceptor model (Saad)










def conv_oper_multichan(x,kernel_1D):

    spatial_dims = x.shape[-1]
    x_reshaped = tf.expand_dims(x,axis=2)
    kernel_1D = tf.squeeze(kernel_1D,axis=0)
    kernel_1D = tf.reverse(kernel_1D,[0])
    tile_fac = tf.constant([spatial_dims,1])
    kernel_reshaped = tf.tile(kernel_1D,(tile_fac))
    kernel_reshaped = tf.reshape(kernel_reshaped,(1,spatial_dims,kernel_1D.shape[0],kernel_1D.shape[-1]))
    kernel_reshaped = tf.experimental.numpy.moveaxis(kernel_reshaped,-2,0)
    pad_vec = [[0,0],[kernel_1D.shape[0]-1,0],[0,0],[0,0]]
    # # pad_vec = [[0,0],[0,0],[0,0],[0,0]]
    # conv_output = tf.nn.conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    conv_output = tf.nn.depthwise_conv2d(x_reshaped,kernel_reshaped,strides=[1,1,1,1],padding=pad_vec,data_format='NHWC')
    
    # print(conv_output.shape)
    return conv_output




@tf.function()#experimental_relax_shapes=True)
def slice_tensor(inp_tensor,shift_vals):
    # print(inp_tensor.shape)
    # print(shift_vals.shape)
    shift_vals = shift_vals[:,tf.newaxis,:]
    shift_vals = tf.tile(shift_vals,[1,inp_tensor.shape[-2],1])
    tens_reshape = tf.reshape(inp_tensor,[-1,inp_tensor.shape[1]*inp_tensor.shape[2]*inp_tensor.shape[3]*inp_tensor.shape[4]])
    # print('tens_reshape: ',tens_reshape.shape)
    shift_vals_new = (inp_tensor.shape[1]-shift_vals[0])*(inp_tensor.shape[-1]*inp_tensor.shape[-2])
    rgb = tf.range(0,shift_vals_new.shape[-1])
    rgb = rgb[tf.newaxis,:]
    rgb = tf.tile(rgb,[shift_vals_new.shape[0],1])
    temp = tf.range(0,shift_vals_new.shape[-1]*shift_vals_new.shape[-2],shift_vals_new.shape[-1])
    rgb = rgb+temp[:,None]
    shift_vals_new = shift_vals_new + rgb
    extracted_vals = tf.gather(tens_reshape,shift_vals_new,axis=1)
    # print('extracted_vals: ',shift_vals.shape)
    extracted_vals_reshaped = tf.reshape(extracted_vals,(-1,1,inp_tensor.shape[2],inp_tensor.shape[3],inp_tensor.shape[4]))
    
    return extracted_vals_reshaped





def generate_simple_filter_multichan(tau,n,t):
    t_shape = t.shape[0]
    t = tf.tile(t,tf.constant([tau.shape[-1]], tf.int32))
    t = tf.reshape(t,(tau.shape[-1],t_shape))
    t = tf.transpose(t)
    f = (t**n[:,None])*tf.math.exp(-t/tau[:,None]) # functional form in paper
    rgb = tau**(n+1)
    f = (f/rgb[:,None])/tf.math.exp(tf.math.lgamma(n+1))[:,None] # normalize appropriately
    # print(t.shape)
    # print(n.shape)
    # print(tau.shape)
   
    return f
    
class photoreceptor_DA_multichan_randinit(tf.keras.layers.Layer):
    def __init__(self,units=1,kernel_regularizer=None):
        super(photoreceptor_DA_multichan_randinit,self).__init__()
        self.units = units
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
             "kernel_regularizer": self.kernel_regularizer
         })
         return config      
               
    def build(self,input_shape):    # random inits
    
        zeta_range = (0.0,0.01)
        zeta_init = tf.keras.initializers.RandomUniform(minval=zeta_range[0],maxval=zeta_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.zeta = self.add_weight(name='zeta',initializer=zeta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,zeta_range[0],zeta_range[1]))
        zeta_mulFac = tf.keras.initializers.Constant(1000.) 
        self.zeta_mulFac = self.add_weight(name='zeta_mulFac',initializer=zeta_mulFac,shape=[1,self.units],trainable=False)
        
        kappa_range = (0.0,0.01)
        kappa_init = tf.keras.initializers.RandomUniform(minval=kappa_range[0],maxval=kappa_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.kappa = self.add_weight(name='kappa',initializer=kappa_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,kappa_range[0],kappa_range[1]))
        kappa_mulFac = tf.keras.initializers.Constant(1000.) 
        self.kappa_mulFac = self.add_weight(name='kappa_mulFac',initializer=kappa_mulFac,shape=[1,self.units],trainable=False)
        
        alpha_range = (0.001,0.1)
        alpha_init = tf.keras.initializers.RandomUniform(minval=alpha_range[0],maxval=alpha_range[1]) #tf.keras.initializers.Constant(0.0159) 
        self.alpha = self.add_weight(name='alpha',initializer=alpha_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,alpha_range[0],alpha_range[1]))
        alpha_mulFac = tf.keras.initializers.Constant(100.) 
        self.alpha_mulFac = self.add_weight(name='alpha_mulFac',initializer=alpha_mulFac,shape=[1,self.units],trainable=False)
        
        beta_range = (0.001,0.1)
        beta_init = tf.keras.initializers.RandomUniform(minval=beta_range[0],maxval=beta_range[1])  #tf.keras.initializers.Constant(0.02)# 
        self.beta = self.add_weight(name='beta',initializer=beta_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,beta_range[0],beta_range[1]))
        beta_mulFac = tf.keras.initializers.Constant(10.) 
        self.beta_mulFac = self.add_weight(name='beta_mulFac',initializer=beta_mulFac,shape=[1,self.units],trainable=False)

        gamma_range = (0.01,0.1)
        gamma_init = tf.keras.initializers.RandomUniform(minval=gamma_range[0],maxval=gamma_range[1])  #tf.keras.initializers.Constant(0.075)# 
        self.gamma = self.add_weight(name='gamma',initializer=gamma_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,gamma_range[0],gamma_range[1]))
        gamma_mulFac = tf.keras.initializers.Constant(10.) 
        self.gamma_mulFac = self.add_weight(name='gamma_mulFac',initializer=gamma_mulFac,shape=[1,self.units],trainable=False)

        tauY_range = (0.001,0.2)
        tauY_init = tf.keras.initializers.RandomUniform(minval=tauY_range[0],maxval=tauY_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.tauY = self.add_weight(name='tauY',initializer=tauY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauY_range[0],tauY_range[1]))
        tauY_mulFac = tf.keras.initializers.Constant(1000.) #tf.keras.initializers.Constant(100.) 
        self.tauY_mulFac = tf.Variable(name='tauY_mulFac',initial_value=tauY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
 
        nY_range = (1e-5,0.1)
        nY_init = tf.keras.initializers.RandomUniform(minval=nY_range[0],maxval=nY_range[1]) #tf.keras.initializers.Constant(0.01)# 
        self.nY = self.add_weight(name='nY',initializer=nY_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nY_range[0],nY_range[1]))
        nY_mulFac = tf.keras.initializers.Constant(10.) 
        self.nY_mulFac = tf.Variable(name='nY_mulFac',initial_value=nY_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)


        tauZ_range = (0.01,1.)
        tauZ_init = tf.keras.initializers.RandomUniform(minval=tauZ_range[0],maxval=tauZ_range[1]) #tf.keras.initializers.Constant(0.5)# 
        self.tauZ = self.add_weight(name='tauZ',initializer=tauZ_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauZ_range[0],tauZ_range[1]))
        tauZ_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauZ_mulFac = tf.Variable(name='tauZ_mulFac',initial_value=tauZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nZ_range = (1e-5,1.)
        nZ_init = tf.keras.initializers.Constant(0.01) #tf.keras.initializers.RandomUniform(minval=nZ_range[0],maxval=nZ_range[1])  #tf.keras.initializers.Constant(0.01)# 
        self.nZ = self.add_weight(name='nZ',initializer=nZ_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nZ_range[0],nZ_range[1]))
        nZ_mulFac = tf.keras.initializers.Constant(10.) 
        self.nZ_mulFac = tf.Variable(name='nZ_mulFac',initial_value=nZ_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
        tauC_range = (0.001,0.5)
        tauC_init = tf.keras.initializers.RandomUniform(minval=tauC_range[0],maxval=tauC_range[1])  #tf.keras.initializers.Constant(0.2)# 
        self.tauC = self.add_weight(name='tauC',initializer=tauC_init,shape=[1,self.units],trainable=True,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,tauC_range[0],tauC_range[1]))
        tauC_mulFac = tf.keras.initializers.Constant(100.) 
        self.tauC_mulFac = tf.Variable(name='tauC_mulFac',initial_value=tauC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
                
        nC_range = (1e-5,0.5)
        nC_init = tf.keras.initializers.Constant(0.01) # tf.keras.initializers.RandomUniform(minval=nC_range[0],maxval=nC_range[1]) # 
        self.nC = self.add_weight(name='nC',initializer=nC_init,shape=[1,self.units],trainable=False,regularizer=self.kernel_regularizer,constraint=lambda x: tf.clip_by_value(x,nC_range[0],nC_range[1]))
        nC_mulFac = tf.keras.initializers.Constant(10.) 
        self.nC_mulFac = tf.Variable(name='nC_mulFac',initial_value=nC_mulFac(shape=(1,self.units),dtype='float32'),trainable=False)
    
    def call(self,inputs):
       
        timeBin = 1
        
        alpha =  self.alpha*self.alpha_mulFac
        beta = self.beta*self.beta_mulFac
        gamma =  self.gamma*self.gamma_mulFac
        zeta = self.zeta*self.zeta_mulFac
        kappa = self.kappa*self.kappa_mulFac
        tau_y =  (self.tauY_mulFac*self.tauY) / timeBin
        tau_z =  (self.tauZ_mulFac*self.tauZ) / timeBin
        tau_c =  (self.tauC_mulFac*self.tauC) / timeBin
        n_y =  (self.nY_mulFac*self.nY)
        n_z =  (self.nZ_mulFac*self.nZ)
        n_c =  (self.nC_mulFac*self.nC)
        
        # print('tau_z: ',tau_y.shape)
        
        t = tf.range(0,inputs.shape[1],dtype='float32')
        
        Ky = generate_simple_filter_multichan(tau_y,n_y,t)   
        Kc = generate_simple_filter_multichan(tau_c,n_c,t)  
        Kz = generate_simple_filter_multichan(tau_z,n_z,t)  
        Kz = (gamma*Kc) + ((1-gamma) * Kz)
        # print('Kz: ',Kz.shape)
        
        # Kz = Kz[None,0,:]
        # print('Kz_new',Kz.shape)
        
        # print('inputs: ',inputs.shape)
        y_tf = conv_oper_multichan(inputs,Ky)
        z_tf = conv_oper_multichan(inputs,Kz)
        # print('z_tf: ',z_tf.shape)
               
        y_tf_reshape = tf.reshape(y_tf,(-1,y_tf.shape[1],y_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        z_tf_reshape = tf.reshape(z_tf,(-1,z_tf.shape[1],z_tf.shape[2],inputs.shape[-1],tau_z.shape[-1]))
        # print('z_tf_reshape: ',z_tf_reshape.shape)
        
        y_shift = tf.math.argmax(Ky,axis=1);y_shift = tf.cast(y_shift,tf.int32)
        z_shift = tf.math.argmax(Kz,axis=1);z_shift = tf.cast(z_shift,tf.int32)
        
        y_tf_reshape = slice_tensor(y_tf_reshape,y_shift)
        z_tf_reshape = slice_tensor(z_tf_reshape,z_shift)
        # print('z_tf_slice: ',z_tf_reshape.shape)
               
    
        outputs = (zeta[None,None,0,None,:] + (alpha[None,None,0,None,:]*y_tf_reshape[:,:,0,:,:]))/(kappa[None,None,0,None,:]+1e-6+(beta[None,None,0,None,:]*z_tf_reshape[:,:,0,:,:]))       
        # print(outputs.shape)
        
        return outputs




class Normalize_multichan(tf.keras.layers.Layer):
    """
    BatchNorm is where you calculate normalization factors for each dimension seperately based on
    the batch data
    LayerNorm is where you calculate the normalization factors based on channels and dimensions
    Normalize_multichan calculates normalization factors based on all dimensions for each channel seperately
    """
    
    def __init__(self,units=1):
        super(Normalize_multichan,self).__init__()
        self.units = units
        
    def get_config(self):
         config = super().get_config()
         config.update({
             "units": self.units,
         })
         return config   
             
    def call(self,inputs):
        inputs_perChan = tf.reshape(inputs,(-1,inputs.shape[-1]))
        value_min = tf.reduce_min(inputs_perChan,axis=0)
        value_max = tf.reduce_max(inputs_perChan,axis=0)
        
        # value_min = tf.expand_dims(value_min,axis=0)
        R_norm = (inputs - value_min[None,None,None,None,:])/(value_max[None,None,None,None,:]-value_min[None,None,None,None,:])
        R_norm_perChan = tf.reshape(R_norm,(-1,R_norm.shape[-1]))
        R_mean = tf.reduce_mean(R_norm_perChan,axis=0)       
        R_norm = R_norm - R_mean[None,None,None,None,:]
        return R_norm














	
	
	
	
#(Saad)
def get_weightsDict(model):
    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    weights_dict = {}
    for i in range(len(names)):
        weight_name = names[i][:-2]
        weights_dict[weight_name] = np.atleast_1d(np.squeeze(weights[i]))
    return weights_dict
		









#Block functions for the model (Ines)













def conv_block(x, n_filters):
    """Conv2D then ReLU activation"""
    x = layers.Conv2D(n_filters, 3, 
                      padding = "same", 
                      activation = "relu", 
                      kernel_initializer = "he_normal")(x)

    return x










def double_conv_block(x, n_filters):
    """ 2 Conv2D """
    x = layers.Conv2D(n_filters, 3, 
                      padding = "same", 
                      activation = "relu", 
                      kernel_initializer = "he_normal")(x)

    x = layers.Conv2D(n_filters, 3, 
                      padding = "same", 
                      activation = "relu", 
                      kernel_initializer = "he_normal")(x)
    return x













def downsample_block(x, n_filters):
    """conv_bloc, MaxPool and Dropout"""
    f = conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p














def upsample_block(x, conv_features, n_filters):
    """ Conv2DTranspose, concatenate, Dropout, conv_block"""

    x = layers.Conv2DTranspose(n_filters, 2, 2, padding="same")(x)

    x = layers.concatenate([x, conv_features])

    x = layers.Dropout(0.3)(x)

    x = conv_block(x, n_filters)
    return x









#Functions to build the models (Ines)








    

def build_photo_unet_model(size, duration, photo, digit_max=10, depth =2, level=1, chan1_n = 9):
    
    """
    Builds a U-net model with the photoreceptors model, 
    then a convolutional U-net with "depth" blocks on each side.
    
    Args:
    size: int, frame_size of the input
    duration: int, duration of the movies
    digit_max: int, default to 10
    depth: int, number of blocks in the model on each side, default to 2
    level: block at which we take the output (default to 1, but I tried pretraining on level 0, with downsampled labels...)
    chan1_n: photoreceptor model parameter, default to 9
    
    Returns:
    the model
    
    """
    
    n_filters = [10] + [32*2**i for i in range(depth)]
    
    # inputs
    inputs = layers.Input(shape=(size,size,duration))
    
    #1st version
    #inputs = layers.BatchNormalization()(inputs)
    
    if photo :
        # Saad's photoreceptor layer

        y = Reshape((inputs.shape[-1],inputs.shape[-3]*inputs.shape[-2]))(inputs)

        y = photoreceptor_DA_multichan_randinit(units=chan1_n,kernel_regularizer=l2(1e-4))(y)

        y = Reshape((1,inputs.shape[-3],inputs.shape[-2],chan1_n))(y)
        y = y[:,0,:,:,:]      

        inputs_unet = Activation('relu')(y) 
    
    else:
    
    	inputs_unet = inputs
    
    #2nd version  
    norm_inputs = layers.BatchNormalization()(inputs_unet)
    f_list, p_list = [], [norm_inputs]
    
    for i in range(depth):
        f, p = downsample_block(p_list[-1], n_filters[i])
        f_list.append(f)
        p_list.append(p)
        
    bottleneck = double_conv_block(p_list[-1], n_filters[depth])
    u_list = [bottleneck]
    
    for i in range(level+1):
        u = upsample_block(u_list[-1], f_list[-i-1], n_filters[depth-i-1])
        u_list.append(u)
        
    outputs = layers.Conv2D(digit_max+1, 1, 
                            padding="same", 
                            activation = "softmax")(u_list[-1])
                            
    
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model
    










# Data generator code from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly (Ines)
#I ended up not using it because I had too many errors with the PR model...












class DataGenerator_frames(tf.keras.utils.Sequence):
    
    def __init__(self, data,labels,
                 n_channels=1,
                 n_classes=11, 
                 batch_size=64, 
                 n_digits = 5, 
                 upsample=True, frame_size = 280, 
                 downsample=False, pool_size=7, strides=7, final_size=4,
                 movie=False, duration=5,
                 digit_max=10,
                 nb_batches = 500,
                 shuffle=True
                ):
        
        
        #'Initialization'
        
        self.n_channels = n_channels
        self.n_classes = n_classes  
        self.batch_size = batch_size
        
        self.n_digits = n_digits
        #n_frames = batch_size
        self.upsample=upsample
        self.frame_size = frame_size
        self.downsample=downsample
        self.pool_size=pool_size
        self.strides=strides
        self.final_size=final_size
        self.movie=movie
        self.duration = duration
        self.digit_max = digit_max
        
        self.nb_batches = nb_batches
        self.shuffle = shuffle
        self.dim = (duration,self.frame_size,self.frame_size)

    
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(self.nb_batches))
    
    def __getitem__(self, index):


        # Generate data
        X, y = self.__data_generation()

        

        return X, y
        
        
    def __data_generation(self):

            
        X,y = generate_frames_dataset(digits_train, labels_train,
                                      n_digits=self.n_digits, n_frames=self.batch_size, 
                                      upsample=self.upsample, frame_size=self.frame_size,
                                      downsample=self.downsample, pool_size = self.pool_size, strides=self.strides,
                                      final_size = self.final_size,
                                      movie=self.movie, duration=self.duration,
                                      digit_max = self.digit_max                                  
                                      )

        return X, y









class DataGenerator_movies(tf.keras.utils.Sequence):
    
    def __init__(self, data,labels,
                 n_channels=1,
                 n_classes=11, 
                 batch_size=64, duration=5,
                 frame_size=280, n_digits=5,
                 depth=2, level=1,
                 
                shadow = False, shadow_ratio = 0.5, light_intensity = 0.1,
                max_jump=1, max_value=254, speed=1,
                digit_max = 10,

                 nb_batches = 500,
                 shuffle=True
                ):
        
        
        #'Initialization'
        
        self.n_channels = n_channels
        self.n_classes = n_classes  
        self.batch_size = batch_size
        
        #n_movies = batch_size
        self.duration = duration
        self.frame_size = frame_size
        self.n_digits = n_digits
        self.depth = depth
        self.level = level
        self.shadow = shadow
        self.shadow_ratio = shadow_ratio
        self.light_intensity = light_intensity
        self.max_jump = max_jump
        self.max_value = max_value
        self.speed = speed        
        self.digit_max = digit_max
        
        self.nb_batches = nb_batches
        self.shuffle = shuffle
        self.dim = (duration,self.frame_size,self.frame_size)
    
    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(self.nb_batches))
    
    def __getitem__(self, index):


        # Generate data
        X, y = self.__data_generation()
        

        return X, y

        
        
    def __data_generation(self):

            
        X,y = generate_movie_dataset_5(digits_train, labels_train,  
                                             n_movies = self.batch_size, duration = self.duration,
                                            frame_size = self.frame_size, n_digits = self.n_digits,
                                            depth = self.depth, level = self.level,
                                            shadow = self.shadow, shadow_ratio = self.shadow_ratio, 
                                             light_intensity = self.light_intensity,
                                            max_jump=self.max_jump, max_value=self.max_value, speed=self.speed,
                                            digit_max = self.digit_max
                                                      )

        return X, y











#Useful function to put arguments in the python command then get them in the script (Richard)
def read_args(data):
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            try:
                name, value, type_var = arg[2:].split("=")
            except:
                raise ValueError(f"Unknown variable format {arg}")	

            if name in data:

                if type_var=="bool":

                    value= value=="True"
                elif type_var=="int":

                    value=int(value)
                elif type_var=="float":

                    value=float(value)

                data[name] = value
            else:
                raise ValueError(f"Unknown variable {name}")
        else:
            raise ValueError(f"Unknown variable format {arg}")
