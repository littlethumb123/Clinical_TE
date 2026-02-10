### Model overview and new components
Before I discusse the changes we proposed; 
I'd like to give a quick refresh on the current version what it looks like, how it works so that you could tell the new pieces where they come from.
The current model has a hierarchical structure, the model first process the codes for every single day and they were aggregated to a single piece of vector and get passed to a series of encoder layer to capture the temporal patterns across days. final goal of this system is to predict the next day's codes for every single member. The embedding we are using is an intermediate output to the end of system and we believe that it should encodes members claim history in past 200 days. 

We preserve the entire hierarchical backbone two reasons. reflects how claim is constructed, organized and processed on everyday basis; and also widely used in EHR modeling. but it is also the fact that this was designed a couple of years ago; considering we are including more data sources and more advanced archtiecture design or components are available; there are opportunities to improve both the computational efficiency and performance. 

First, the training cost and computational sources constraints are always a big headache. a culprit is that computation in the current model is not efficient to GPU and therefore it takes longer and generate higher cost for training and inference. 
Here is why, the core of a transformer is something called self-attention score; you could simply think of it as a matrix of relationship score between codes; like which is semantically closer to which codes. which is more clinical indicative of which codes. However, the calculation of such score in GPU is extremely memory intensive when the matrix is huge. 
What we did differently is using someting called flash attention; it chunks the big matric into smaller pieces and speed up the calculation by condensing it into a fused kernel; this provides faster training and inference with lower GPU requirements; this is espeially important for teams with limited GPU resources

Second, we tried to improve the training and optimization strategies. this is a suite of modifications to improve the training quality, stability reduce the GPU memory required given the new data and also given the resoures constraints. 
I am not gonna expand on this due to the time concerns; but we will be more than happy to provide more details if you may have any questions in QA or after the meeting

These two modifications are considered more from engineerng perspective; 
The third one is more from embedding quality perspective we tried to improve the model's capability of capturing the heterogeneity of the data. 
the idea is driven by some interesting findings in our last thinkubator where we found this model is accumulating knowledge over and over layers, and inteterestingly it cannot tell the difference between clinical concepts, such as diabetes, depression, hypertension and sdoh until the layer 3, 4 and 5. 
This is an important signal for us because we saw the model is trying to be expertised in processing different informaiton; so we wonder if it is possible to twist the model help the TE to be more specialized an expertised to better process different aspects of the data and have a more representative embedding for each member. we used something called mixture of experts; what's happening is instead of having a single layer to process all of the codes; we have the model to route different codes to the layer that is specialized in processing its informaiton. ideally like diabetes code to diabetes expert, depression codes to depression. this is just a simplied and imprecise analogy but hope you get the idea

### Model experimentation results
In order to make sure the modifications are effective and do not introduce any unnecessary complexity and computational cost or even impact the performance. We set up a series of ablations.
the data we are using 1.75M sampled members across 3 LOBs in 2023; the members with at least 10 days of medical activtiies over the past 36 months were included. 90day claim finalization window is applied to the data to make sure the data is complete and fully adjudicated. The reason we use sampled population is to have quick turnaround for informed decision making which changes should be applied given the limited GPU resources. 


If you look at the table; on the left hand side the three different modfications we applied incrementally to the current model so that we can compare the effectiveness one by one with only one variable changed at a time. the metrics we are using including the prediction performance and training time and GPU memory usage. the recall@10 is how many of members having at least one code predited correrctly in the top10 predictions; the precision is among the top10 predictions, how many of them are true. the micro-recall is across all true codes, how many of codes are predicted correctly in the top10 predictions. the NDCG (Normalized Discounted Cumulative Gain) is a ranking quality metric to see if the correct codes are near the top10 predictions, a score of 1 is the best. We also measures the total amount of time required for training and peak GPU memory usage during training to track efficency.
The top row is the random guess on over 6000 codes; they are all near zeros as we expected. 
The second row is the current model as a baseline
the third row is where we optimized the training srategies to the model; the performance is nearly 1.5 times better than the current model; this is what we expected; 
The fourth row is where we tried to speed up the training with the flasth attention, the performance nearly identical to the current model with optimzied training but the training cost is reduced by 34% and GPU memory usage is reduced by 61%. this is also what we expected. 
The final row is where we tried MOE, it's also nearly identical to the optimized with a slight increase in cost and GPU memory usage. 
these ablations are intrinsic and preliminary; but they are going to inform our next downstream evaluation; where we take the model to real world and see how the embedding actually performs in the inpatient models across LOBs. One important note is that A 2× improvement in pretraining micro-recall does not imply a doubled improvement on downstream tasks—the relationship should positive but non-linear. So we wouldn't expect a see the same magnitude of improvement. 

Some lessons learnt we'd like to share; the use of flashattention shows big lift on training efficiency and potentially in inference. We believe the real promising is the scalability: the benefits grow with sequence length. This opens door for us to leverage longer patient histories, like 3-5 years instead of 200 days in the future, without exponentially increased computational cost.

another thing we wanna share is that the GPU resources are limited on GCP across enterprise; for any pretraining or finetuning works, the GPU resource planing needs to be part of the project scoping at day 1,  a cost estimation, governance approval and plan and reserve compute resource for more advanced GPU ahead of time is a must for quick start and success. 
The infrastructure shapes what's technically feasible, so also get ready for walkaround solutions; we have made several adaptations on the model architecture to make it work with exisiting infrastructure. 










### 5-min version
We have made a few modifications to the model architecture to improve the computational efficiency and performance; and set up a series of ablations to gauge their effectiveness and make sure they do not introduce any unnecessary complexity or computational cost. The ablation used 1.75M sampled members across three LOBs. we applied a few criteria, the member should have at least 10 days of medical activity over 36 months. We applied a 90-day claim finalization window to ensure data completeness and fully adjudicated.

If you look at the table; we are using ranking metrics for evaluation; the recall@10 is how many of members having at least one code predited correrctly in the top10 predictions; the precision is among the top10 predictions, how many of them are true. the micro-recall is across all true codes, how many of codes are predicted correctly in the top10 predictions. the NDCG (Normalized Discounted Cumulative Gain) is a ranking quality to see if the correct codes are near the top10 predictions, a score of 1 is the best. We also measures the total amount of time required for training and peak GPU memory usage during training to track efficency. 
The top row is the random guess on over 6000 codes; they are all near zeros as we expected;
the current model is the one we replicates original architecture and configurations as closely as possible;
The third row is where we applied a suite of optimized training strategies to the current model architecture to better predict rare codes; stabliizng training converge procedure, improving GPU memory usage; normally a must when we have new data, new design, we need to tune the configurations. this set up nearly doubled the performance than the current model and configurations; this is what we expected, we have to adapt TE for the new data. 

The fourth row is where we tried to speed up the calculation of self attention by using something call flash attention. Quick context: the bottleneck in any transformer is self-attention, which requires moving massive matrices between GPU memory and compute units. Flash attention solves the computation by chunking the computation and fusing it into a single kernel which is more GPU friendly and reduces the memory usage by over 60% and reduce the training time by 30% than the current model.

The final set up is where we tried to improve the quality of embeddings in representing heterogeneity of the data; the idea is driven by last year thinkubator where we found that the transformer started to learn different clinical concepts, such as depression, hypertension diabetes, from the second layer all the way to the final; so we were thinking if it is possible to help TE to learn to be specialized; we replace the feedforwrad neural network with soemthign call mixture of experts; how it works is during the inference all of the tokens no longer pass through a single dense model, instead, each token will be routed to their favorite sub-networks to be processed; this is what we hope such design could improve specializaiton with nearly zero cost. as you could see the performance roughly identical to the current architecture with little increase in time and GPU usage. 

all of these observations are intrinsic and preliminary; and going to inform our next downstream evaluation; that is where we take the model to real world and see how the embedding actually performs in IP model across LOBs. 
One important note is that the doubled improvement in pretraining recall does not imply a doubled improvement on downstream tasks; the relationship should positive but they are on-linear. So we wouldn't expect a see the same amount of improvement in the downstream evaluation. 



QA: 
- Why using ranking metrics: 
Because of extreme class imbalance and clinical use case alignment. With ~6,000+ possible codes and only ~5-20 true codes per patient-day, traditional multi-label metrics are dominated by true negatives and don't reflect clinical utility.

- What is included in a layer 



- What is the reason why MOE is not better than the legacy model?
    - Theoretically 