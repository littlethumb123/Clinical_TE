### Model overview and new components
Before I discusse the the architecure we proposed; 
I'd like to give a quick refresh on the legacy version what it looks like, how it works so that you could tell the new pieces where they come from.
The legacy model has a hierarchical structure, if you look at the diagram on the left hand side, from bottom to top. it takes features like age, gender everyday's codes; and in the first level it processes the codes within each day for a member; then aggregated together as a single vector going through a series of encoder layers to capture the temporal patterns across days; and final goal is to predict the next day's codes for a each member. The embedding we are using is an intermediate output to the end of the temporal encoders. 

Now, this architecture was designed a couple of years ago and we are including more data sources and more advanced archtiecture design or components are available; there are opportunities to improve both the computational efficiency and performance. 

First of all computational efficiency. the core of the entire transformer model is something called self-attention; it captures the relationship bewteen codes, between different days; the bottleneck is that this is an extremely memory intensive operation inside GPU. A GPU has two parts, one part for data storage and one part for computation; just like the hard drive vs. CPU in our laptop; and the calculation happens during the back and forth between these two parts; the traditional way to calculate the relation scores is moving a very large matrix from one place to another; this is not what GPU likes. what we did differently is someting called flash attention; it chunks the matrics into smaller pieces and speed up the calculation by condensing it into a fused kernel; this provides faster training with lower GPU requirements; this is very important for teams with limited GPU resources

Second, we tried to improve the training and optimization strategies. this is a suite of modifications to improve the training quality and stability given very imbalanced and long-tailed distributed data from different sources and also improve the GPU memory usage given the constraints of GPU resoures we are using. I will leave this for QA for time constraints if you may have any questions. 

These two modifications are more from engineerng perspective; 
The third one is more from embedding quality perspective and thus more exploratory and experimental

we are trying to improve the way the information is encoded to the embedding. the idea is driven by the some outcomes in our last thinkubator;  by adding a little bit specialization to the model; that is,  we try to have model to process tokens in a more specialized way with a hope that the final emebdding can better reflect the heterogeneity of the data. the idea comes from last year thinkubator when we were trying to interpret how the transformer processed information, we found that the transformer started to learn different clinical concepts, such as depression, hypertension diabetes, from the second layer all the way to the final; 
we replace the feedforward neural network with a mixture of experts; this is a component that improves the model specializaiton without increasing model size and inference cost. during the inference each token or the code will be routed to two favorite dense layers to be processed. 

### Model experimentation results
In order to make sure the modifications are effective and do not introduce any unnecessary complexity and computational cost. We set up a series of ablations.
the data we are using to pretrain the TE is a 1.75M sampled members across 3 LOBs with at least 10 days of medical activtiies over the past 36 months. 90day claim finalization window is applied to the data to make sure the data is complete and fully adjudicated. The reason we use sampled population is to have quick turnaround for the changes we are going to make given the limited GPU resources. 
Now the results; 
The changes are applied incrementally from top to bottom; starting with the legacy model which replicate the existing model architecture and configurations as closely as possible; then apply the optimized training strategies to the legacy architecture; compare these two, we want to see if the optimized configurations make sense; if you look at all performance metrics, the recall, precision and micro-recall at top 10 predictions are roughly doubled and GPU usage is a little reduced; 
Then compare the third one with fourth one which one uses flash attention and one does not. the performance are roughly identical but the training time and GPU usage get reduced by 50%. 
All of these are what we expected to see. 
Now the interesting part comes to the MOE; it didn't bring at least 2-5% performance lift we expected at the beginning. The reason is multifaceted
1) model size; The MOE is usually used to serve billion level LLM, while our model is only 27M parameters; the downside of a small model is smaller embedding dimension, smaller hidden dimension, the experts don't have enough room to learn and specialized; 
2) The routing is hard to stablize the training: small model has small batch_size, that means the experts can see way fewer tokens than in a larger model during training; this means he informaiton is high variance and noisy and hard to learn
3) data diversity; the clinical claims compare to natural language is more homogeneous; this added another layer of difficulty for expert to be specialized. 
Simply put, it's like you put a V12 engine on a honda civic. no means to be offensive to civic owner but the overhead outwighs the benefits.

However, all of these ablations are intrinsic, preliminary and going to inform our next downstream evaluation; that is where we take the model to real world and see how the embedding actually performs. 

the last lessons we learnt around computational resources. 
the GPU resources are limited on GCP across enterprise; for any pretraining or finetuning works, the GPU resource planing needs to be part of hte project scoping at day 1 to do a cost estimation, governance approval and plan and reserve compute resource for more advanced GPU; 
The infrastructure shapes what's technically feasible and get ready for walkaround solutions; we have made several adaptations to the model architecture to make it work with T4. 



