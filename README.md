# Deep Learning and Practice_conditional GAN
 
## Introduction
The target of this assignment is using conditional GAN to generate a sequence of pseudo images based on given object labels.
GAN basically contains 2 models: discriminator and generator. Discriminator classifies whether given images are real or returns the degree of authenticity and generator try to produce those pseudo image which are real enough to cheat discriminator. Two models train alternately and finally we can get a generator which has the ability to produce quality pseudo images.
	Dataset
There are totally 24 objects in i-CLEVR datasets with 3 shapes and 8 colors and the object ID is from 0 to 23. Each sample image contains 1 to 3 objects. The account of training dataset is 18009 and the test dataset has 32 groups of object labels.

## Implementation details
-	Hyper Parameters Setting
Variable name	Value or Type
Iteration	epoch	30000
	Iteration for D	4
	Iteration for G	1
Training batch size	64
Learning rate	Learning rate for D	0.0001
	Learning rate for G	0.0003
Optimizer	type	Adam
	Beta1	0.5
	Beta2	0.999
Loss function type	hinge
Trick	Add noise	True
-	Data Loader
For training, data loader will return image and label id list with length 3. Image is transformed from PIL image to tensor and label id list is expanded to length 3 by additional class {24} which means “null”.
For testing, data loader returns label id list after same process and the respective one-hot code for evaluation.
-	Model structure
I applied the technique of “SNGAN and cGAN with Projection” as my conditional GAN structure. This structure projection the conditional information (label) into both discriminator and generator.
Both discriminator and generator base on Resblocks.
 
### Generator
  
	Discriminator
It returns a score of given image and labels. Higher score infers to more likely being real, and lower score means that the image is more fake.
 
 
 
	Loss Function
For generator:
Loss_G=-mean(D_fake)
 
For discriminator:
Loss_D=mean(1-D_real )+mean(1+D_fake)
 
	Trick in training
I test this trick in training, but only take “adding noise” in my best result in the end.
	Adding noise
Adding noise into model input can help the model training. Sometime the GAN broken because the generative distributions are weird. Adding noise into training can help the model close to common distribution.
 
	Flip label
This trick is to swap the binary label from 0 to 1 or from 1 to 0. This skill can confuse the discriminator and avoid the failure of too strong discriminator. 
In experiments, flipping label will cause the generator produce same object to fool discriminator.
Result of original test after iteration 4000 and 8000
  

	Relativistic loss
This idea comes from making the score of generated images closer to real data score. Therefore, it calculates the “distance” between scores of real and fake images. Unfortunately, it brings out more terrible results after training.

	Decay learning rate
To avoid swinging, I modified learning rate within training and start decay after 4000 iterations.
 
	Smooth label
If using binary label such like True and False. It is more suitable for smooth label instead of discrete label. For example, if the sample is true, make label change to 0.8~1.2 from 1; otherwise, change to 0~0.2 from 0 if it is a fake sample.
I only use this trick in DCGAN structure. In final structure, I didn’t use binary label in computation.
 
 
	Results
	Best result
Original test score=0.791667 (iteration 37000)
 
New test score= 0.821429 (iteration 37000)
 

	Observation from Experiment Process
Followed is using original test data to generate pseudo images within training.
The generated image can show blurred objects since iteration 500, and score starts to reach 0.22.
 
The generated object become clearer after iteration 5000.
 
The quality of generated image keeps increasing. Bellowed images are the result after iteration 15000.
 
	Discussion
	cDCGAN vs. cGAN with projection
The result of cDCGAN is poorer than using projection structure. The most different element is the application of label. Using projection embed label information and project them in the output of discriminator output and combine the latent code from label in each layer of generator. In other hand, cDCGAN just “concat” image and label before throw them into network.

	Sum label or Using Linear layer
Because our samples have multiple labels in one image, we need to embed them and do reshape to match layer input size. I try “sum” them and transfer by linear layer. The results do not have significant difference.
	Proportion between the Iteration of Discriminator and Generator
In training, we should guarantee the ability of discriminator to lead generator training. However, if discriminator the generator will give up learning which means model collapse. Overall, I train discriminator more times than generator with lower learning rate to control the balance.
There are some model try to deal with the balance problem, but I did not try those method in this assignment.

	Reference
SNGAN and cGANs with projection discriminator
How to Train a GAN? Tips and tricks to make GANs work
