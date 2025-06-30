# Part 1: Efficient Inference - Pruning and Sparsity
![alt text](Images/image-1.png)

## Today's AI is too BIG!
![alt text](Images/image.png)

## MLPerf
It is a competition participated by many companies, which focuses on optimizing models while maintaining accuracy.
![alt text](Images/image-2.png)
- Closed Division: The model must not change (precision, #Weights, etc). Purely using hardware optimization.
- Open Division: Allowed changing the model (prune model, quantize, compress, etc) as long as accuracy is maintained.

### MLPerf: Open division
![alt text](Images/image-3.png)

## Motivation for Pruning: Memory is expensive
![alt text](Images/image-4.png)
- DRAM consumes a lot more energy, we want to reduce access to DRAM for weights and activations.

## Section 1: Pruning

### Introduction to Pruning
Removing synapses and neurons.
![alt text](Images/image-5.png)
![alt text](Images/image-6.png)

Fun fact: Pruning also happens in the human brain as we grow older:
![alt text](Images/image-7.png)

### Neural Network Pruning:
1. Pruning can lead to test accuracy drop increases as pruning ratio grows.
![alt text](Images/image-8.png)

2. Finetuning after pruning can recover the accuracy, sometims even increase the accuracy by reducing the overfitting.
![alt text](Images/image-10.png)

3. With iterative pruning and finetuning (doing the step 1 and 2 multiple times) we can push the drop to the right hand side even further, allowing high pruning ratio without accuracy drop.
![alt text](Images/image-11.png)

Pro tips:
- The professor suggests that with iterative pruning, we should gradually increase the pruning ratio each iteration (eg. 10% -> 20% -> 40%) and finetune the pruned model, instead of directly pruning with high ratio such as 80% in the first iteration.
- Usually there is a threshold for the pruning ratio, beyond the threshold it will be difficult to recover the accuracy.

Larger models have more room for reduction:
![alt text](Images/image-12.png)

Visualization of Pruning results:
![alt text](Images/image-13.png)
- Even with 90% pruning, the model accuracy does not degrade, however the threshold is 95%.

Publication history of pruning:

![alt text](Images/image-14.png)

Industry adoption of Pruning:
Hardware that supports sparse matrix multiplication can save compute for sparse networks because anything multiplied by 0 is 0, we can ignore them. This requires specialized hardware that has sparse matrix kernels.

![alt text](Images/image-15.png)
![alt text](Images/image-16.png)

### How to formulate Pruning:
Learning the notations is useful for when reading and understanding papers.
![alt text](Images/image-18.png)
- After pruning, the number of non zero weights should be less than the target number of non zero weights, N.

## Granularity of Pruning:
Pruning can be performed at different granularities, from structured to unstructured. Prune by 4x4  matrix or 2x2 matrix, etc.
![alt text](Images/image-20.png)
- Unstructured pruning: Happens randomly
- Structured pruning: Follows a pattern, the result is still a dense matrix which we can use with most hardware and libraries.

### Pruning Convolutional Layers 
Recap:

![alt text](Images/image-21.png)
- We have 4 dimesions to prune/can be pruned.

Spectrum of pruning

![alt text](Images/image-22.png)
- From completely random (left) to following a strict pattern (right).

#### Fine-grained Pruning

1. More flexible
- ![alt text](Images/image-23.png)

2. Larger compression ratio
- ![alt text](Images/image-24.png)

3. Can deliver speed up on some custom hardware (e.g. EIE) but not GPU (easily).

#### Pattern-base Pruning
Nvidia has been using pattern based pruning.
![alt text](Images/image-25.png)

Question: What is the pruning pattern for the matrix above?
<details><summary>Answer</summary>For every 1x4 vector, 2 weights will be zeroed. Therefore the sparsity ratio is 50%. The benefit of this is we can compress them back into a dense matrix with half #Weights. Some overhead to store the indices, only 2 bit required for 4 combinations so it is quite small.</details>


Accuracy results:
- ![alt text](Images/image-26.png)

#### Channel Pruning
The most popular method. We can get very dense matrix, which is easy to accelerate. The downside is the pruning ratio would be lower. This is the most common techniques for mobile chip companies.

![alt text](Images/image-27.png)

We can perform uniform shrinking (every layer has the same ratio) vs Channel pruning where each layer has different pruning ratios:
![alt text](Images/image-29.png)

Latency vs accuracy tradeoffs"
![alt text](Images/image-30.png)
- Wisely choosing the pruning ratio for different layers is better than uniformly pruning with the same ratio across layers.

## Pruning Criteria
What synapses and neurons should we prune?

Question: Using the neuron below, which weight should we remove? Use your own intuition.
![alt text](Images/image-31.png)
<details><summary>Answer</summary>We should remove the weight 0.1, because it has the least impact towards the output y. This is called Magnitude-based Pruning, it should be L1-Norm ed before selecting the magnitude, 2 is less important than -10.</details>

### Magnitude-based Pruning
Element wise pruning:
![alt text](Images/image-32.png)
- This is fine-grained pruning

Row wise pruning:
- ![alt text](Images/image-33.png)

OR with L2-norm:
- ![alt text](image.png)
- This is similar to vector based pruning

Note: These simple techniques have been used in industry successfully. Sometimes easier techniques will make it into products instead of complex ones.

### Scaling-based Pruning
![alt text](Images/image-36.png)
- We can learn the scaling factor if we are doing Channel Pruning

We can actually just reuse the scaling factor computed during the batch normalization layer.
![alt text](Images/image-37.png)

### Second-Order-based Pruning
Pruning based on optimizing a loss function.
![alt text](Images/image-40.png)
![alt text](Images/image-41.png)
- Hessian matrix is heavy/difficult to compute because of the second order derivative calculation, people try using approximation instead. More self reading on this topic if interested.

## Selection of Neurons to Prune

### Weight Pruning
![alt text](Images/image-42.png)
When a neuron is pruned, all the weights associated with it is pruned, therefore it is called Course-grained pruning. Channel Prunning is similar.

### Activation Pruning: Percentage-of-Zero-Based Pruning
Example to determine which activations to prune. With ReLU, there will be zero activations.
![alt text](Images/image-44.png)
- To do Channel Pruning, calculate the number of zeros, across both batches and take the average. the channel with the most APoZ activations will be pruned. Channel 2 should be pruned because it has large ratio of zeros.

### Regression-based Pruning
Minimizing the error before and after pruning. The Matmul should be close before and after pruning.
![alt text](Images/image-46.png)
- The output matrix before and after pruning will be of the same shape if we prune the channel dimension.

Problem formulation to minimize the reconstruction error.
![alt text](Images/image-47.png)
- Finds the channel which gives the least error after prune.
- When Beta is 0, it means that channel is pruned.
- Beta is learned in step 1. The weights, W is fixed.
- Weights, W is learned in step 2. Beta is fixed. This means adjusting the weights after pruning.
- This can be done iteratively.
- With this technique we do not need backpropagation, instead we just reconstruct the weights for individual layers. Saving tons of GPU compute. 
- The drawback is it does not have a wholistically picture of the entire model.

## Pruning Demo notebook in /Chapter-1/demo-Pruning.ipynb
Colab link: https://colab.research.google.com/drive/1Z3Qne88hrTuojRwigIEyvUK5BCNJXeRZ?usp=sharing#scrollTo=2tFjnZZVlIFL

## Summary of lecture
![alt text](Images/image-48.png)
![alt text](Images/image-49.png)