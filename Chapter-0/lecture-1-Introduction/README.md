# Lecture 1: Introduction

Slides: https://www.dropbox.com/scl/fi/h3ggav4eopxsitqxzf6t2/Lec01-Introduction.pdf?rlkey=hzbpsha72p5e3ed4mdvcgcda5&e=1&st=pz5u977e&dl=0
Video: https://www.youtube.com/watch?v=U7EPZv8Kh9w&feature=youtu.be

Model size grows every year at a much faster rate than GPU memory:
![alt text](images/image.png)
![alt text](images/image-1.png)

Studies on model compressions is becoming more and more popular:
![alt text](images/image-2.png)

### Use cases:

## 1. Vision

![alt text](images/image-3.png)

### On device inference:

![alt text](images/image-4.png)
![alt text](images/image-5.png)

### On device training:

![alt text](images/image-6.png)

- Training is much more expensive than inference, because we would need to store the activations, calculate the gradients, perform backpropagation, etc.

![alt text](images/image-7.png)

- Using quantization and other model compression techniques, training on limited hardware such as microcontrollers is possible.

![alt text](images/image-8.png)

Segment Anything optimization:
![alt text](images/image-9.png)
![alt text](images/image-10.png)

Image generation models optimization:
![alt text](images/image-11.png)
![alt text](images/image-12.png)
![alt text](images/image-13.png)
![alt text](images/image-14.png)
![alt text](images/image-15.png)

Other use cases: 3D image generation, video generation, self driving with 3D perception, sensor fustion, etc

## 2. Language

![alt text](images/image-16.png)

Use cases: Coding, translation, etc

![alt text](images/image-18.png)

- Lite transformers is able to reduce the transformer model from 176MB to 9.7MB, making it much easier to deploy on a phone.

![alt text](images/image-19.png)

- There are different methods to inference an LLM based on the tasks, it is a tradeoff between accuracy and first token latency.

![alt text](images/image-20.png)

- Every percent of accuracy improvements comes at a much larger cost.

![alt text](images/image-21.png)

- Chain of thought prompting includes a thought process in the few shot example in the prompt, forcing LLMs to think before answering complex questions, making it more accurate for reasoning problems.

![alt text](images/image-22.png)

- Again, model size needs to be big to show meaningful improvements with COT

Sparse attention prunes "less useful" tokens (e.g. "I", "the", "a"), improving efficiency while maintaining accuracy.
![alt text](images/image-24.png)

![alt text](images/image-25.png)

- Deploying LLM on the edge also improves latency, removing network latency.

![alt text](images/image-26.png)

- Smoothed quant. levels makes quantization easier, will learn in this course. C/C++ prerequisites is important, manipulating pointers, SIMD, cache, locality, multithreading, multi core processors, register, parallelism.

![alt text](images/image-27.png)

![alt text](images/image-28.png)

## Multimodal

Combining modalities such as Images, Languages, Audios, Videos, Actions, etc.

Example:
![alt text](images/image-29.png)

![alt text](images/image-30.png)

- Image above shows that AWQ quantization gives better accuracy than naive methods.

![alt text](images/image-31.png)

- Visual transformers tokenizes not only texts, but also images! It will then project them into the same vector space for generation.

Example prompt:
![alt text](images/image-32.png)

Visual in-context learning, similar to few-shot prompting:
![alt text](images/image-33.png)

Text to Action modality:
![alt text](images/image-34.png)

AlphaGo and AlphaFold:
![alt text](images/image-35.png)
![alt text](images/image-36.png)

# Three pillars of Deep Learning:

![alt text](images/image-37.png)

Quantization and pruning brings significant improvements:
![alt text](images/image-38.png)

- New trends: Parallel computing, specialized hardware, low precision compute

Software innovation is important:
![alt text](images/image-39.png)

- This course utilize existing hardware, make software more efficient on existing hardware, implications for designing new hardware.

Cloud AI Hardware:
![alt text](images/image-40.png)

- TOPs/compute is growing fast.
- Memory bandwiwdth is more expensive, energy is dominated by moving data.
- Unfortunately power is also growing fast, an 8 GPU node will take 4 to 5 thousand Watts, energy and cooling will be the bottleneck. Now two cables can only serve 4 nodes of A100 and 2 nodes of H100, previously it can serve an entire rack.
- Memory growing slower than compute.

Edge AI Hardware:
![alt text](images/image-41.png)
![alt text](images/image-42.png)

- Performance improving fast as well. However the graph shows peak performance, which doesn't indicate speed accurately, speed can be affected by activations, data movement, utilization, etc.
- Power remains relatively flat.
- Memory has steady improvements.

Mobile GPU, used in EVs:
![alt text](images/image-43.png)

Microcontrollers, used in IoT devices such as cameras:
![alt text](images/image-44.png)

Big Gap
![alt text](images/image-45.png)

- We can close the gap between Cloud AI and Edge AI by using model compression techniques.

Current ML vs Tiny ML
![alt text](images/image-46.png)
![alt text](images/image-47.png)
