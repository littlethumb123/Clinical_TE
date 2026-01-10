### Model overview and new components
Before I discusse the the architecure we proposed; 
I'd like to give refresh on the legacy version so that you can tell where each new pieces comes from.
This is a conceptual diagram of the model architecture; on the left hand is the backbone; it is designed in a herarchical structure where it first processes each member's claim codes in a daily basis and then aggregate them and pass them to a series of encoder layers to capture the temporal patterns; this is trying to adapt to what the claim looks like and how it's processed by human in real life and also validated by a few prior research
The entire model is trained to predict each member's next day's codes based on the history; so it's a multilabeling with 6297 target codes groups; the embedding is an intermediate output between the final temporal layer with final output layer; what it really represent is a patient history summary of the past 200 days. 

Now here are three major points we tried to improve from model architecture perspective; (3min) 

1) Faster training and (potential) inference;
- the legacy model uses self-attention to capture the relationship between codes and between different days; this has been found to be extremely memory intensive inefficient for GPU. The GPU has two major parts, one for data temporary storage, just like the hard disk in your laptop and a place for fast computation, just like CPU, the pain point is when the relation between tokens or beteen days is calculted as a big matrix; there will be a huge size of data transaction between those two parts; this is not what GPU likes; what we use instead is something called flash attention; it chunks the computation of the matrics into smaller pieces and speed up teh computation by condensing it into a fused kernel; this has been widely validated and used in the industry for large language model. 

2) Second improve the training and optimization strategies
these modifications are made for the purpose of improving prediction of the rare codes, stablizing the training converge process and optimzie the GPU memory usage given the constraints of T4 GPU we used. 
The these two modifications is primarily planned for engineering efficiency and training quality

3) better embedding representation; we are trying to better the embedding representation by adding a little bit specialization to the model; that is,  we try to have model to process tokens in a more specialized way with a hope that the final emebdding can better reflect the heterogeneity of the data. the idea comes from last year thinkubator when we were trying to interpret how the transformer processed information, we found that the transformer started to learn different clinical concepts, such as depression, hypertension diabetes, from the second layer all the way to the final; 
we replace the feedforward neural network with a mixture of experts; this is a component that improves the model specializaiton without increasing model size and inference cost. during the inference each token or the code will be routed to two favorite dense layers to be processed. 

### Model experimentation results
For the experimentations; We have a series of setups and experiment with each of them for a couple of rounds and applied the modifications incrementally. Due to the constraints of GPU resources and time consuming training procedure, we can only train it with 1 epoch and 10% of the entire population to experiment as many setups as we could.
Now we ended up with four versions
the one that exactly replicate the current model architecture; 
the one that keep the backbone and implement the optimized training strategies
the one that keep the backbone and impelement flash attention and optimized training strategies
teh one implemented MOE

The metrics we are using are relatively intrinsic and primarily indicates about the training performance of predicting next days' sequence. 
Recall 

there are two stage evaluations in our plan, first is to evaluate intrinsic performance of the model; that is how accurate it predicts the next day's codes; how long it takes 


### Model experiment results
1. we have designed a series of experimentations to evaluate the efficiency and performance lift of those three primary improvements; 



### Lesson learnt
1. mis-scale; the MOE was originally designed for billion parameters while the current model is only 27M parameters; it's like putting V12 engine on a honda civic. overhead associated with the routing mechanisms outweighs any potential advantages,
2. small dimensions: the router that routes the tokens must learn to differentiate which expert should process; only 256 dimension does not provide sufficient variance for learning. 
3. data itself, the clinical concepts are inherently homogeneous and it's not like natural language. 
4. Hard to tune; a critical issue is expert collapse; the router is routing most of the tokens to very few experts and other experts died out and learn nothing; this will collapse to a single expert. there could be a lot of factors, 
    - router initialization: small perturbation at teh beginning can get amplified and favor one of two of experts; then these expert start to dominate. 
    - loss over regularization