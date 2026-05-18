Hi everyone. our team has completed the Transformer V3 retraining and downstream evaluation, but we have not got chance to deep dive and explained how the model works or how to explain the embeddings. And also also there are a series production related updates we So today I want to walk through the architecture end to end and share a toolkit and a framework we have explored to explain these embeddings in downstream classification work.

Slide 3:
This diagram shows the full end-to-end transformer architecture. At a high level, the model has a hierarchical structure, it first addresses the claim codes at daily level and then captures temporal characteristics across days. is trained to predict which clinical codes will appear on the next day for each member. The 256-dimensional embedding is the intermediate representation produced right before that prediction.

I will walk through this bottom up, in the same direction the data flows.

For each member, the model starts with codes observed on each day. We keep up to 80 codes at maximum per day, and each code has its own embedding with a dimenion of 256. Those code embeddings then go through learned attention pooling. In practice, this is a single self-attention layer that learns which codes matter most within that day.

Originally, this part of the model was a one-layer transformer with max pooling. We replaced it because learned attention pooling is about twice faster in daily encoding while achieving the same performance as the old design. After that step, the 80 code vectors are collapsed into one daily vector, and then demographic context is added.

At that point, each day is represented by a single vector, and each member has up to 200 daily vectors because we look back across 200 days of claim history.

Next comes the six-layer temporal encoder. This is where the model learns how a member's history evolves over time.

Each layer has the same core structure. The first part is flash attention with a causal mask. I will come back to the causal mask in a moment. Functionally, it has the same role as a regular multi-head self-attention: each day looks back at earlier days and decides which past events matter most. but it uses GPU memory much more efficiently. So we got faster training without giving up predictive quality.

The second part is type feed forward neuro-network, Swish-Gated Linear Unit, or SWIGLU, the basic function is for non-linear transformation, but SwiGLU adds a gate that helps the model decide what information should pass through. We used SWIglu to help stabilize training and improved convergence speed

Now let me move back to the main route and talk about how the model learns the temporal features. If you still remember we have the raw matrix of 200 days ready to be processed. this matrix incldues all information about one's claim history and demographic context. 
When the matrix enter the layer 0, for every single day vector the model is asking: who you are and what your relationship with your past days
let's take an example, When day 3 enters layer 0, the model is effectively asking: how is day 3 related to day 1 and day 2? If day 3 contains an acute event, the model may learn that a diagnosis pattern or medication use from two days earlier is highly relevant. So mathematically, the output for day 3 is no longer just the original day 3 vector. It becomes a blend of its own information and the most relevant signals from earlier days.

In layer 1, the same process happens again, but now the inputs are no longer raw day vectors. They are already contextualized outputs from layer 0. This is where the abstraction becomes deeper. By this point, the day 3 vector is not just combining day 1, day 2, and day 3. It is combining layer 0's interpretation of those days.

The simplest way to think about it is this: layer 0 learns the relationships among days, and layer 1 learns from what layer 0 has already learned.

By the higher layers, each day vector becomes a compact clinical state that summarizes the most relevant history up to that day. The day position itself does not move, and the vector stays 256-dimensional, but the information inside it becomes more contextual and selective.

At the top, the model makes a multi-label prediction of which grouped clinical codes will appear on the next day given the prior history. The member embedding is taken immediately before that output layer, using the representation at the last day. if a member has 200 days of history, we then take the output vector at the day 199 position at the layer 5. 


Slide 4:
One major lesson from this work was our experience with mixture-of-experts, or MoE, inside the transformer. I happpened to learn some folks here are also exploring LLM-style ideas for other problems, so I wanted to share what we learned and hope it could be helpful as data point when you are thinking about architecture options

MoE is now a standard scaling idea in LLM. The idea is intuitive, Instead of sending every token through the same feedforward block, the model uses a router to send each token to a small number of experts. During inference, only those selected experts are active, so you get more model capacity without paying the full dense-compute cost every time. In some setups, you also add a shared expert to capture common patterns while the other experts specialize.

The motivation for trying this was straightforward. Clinical populations are heterogeneous. Members differ in disease burden and utilization patterns, so it was reasonable to ask whether MoE could let different experts specialize around different clinical concepts or member archetypes.

I started with 8 experts, then tried 7 experts plus 1 shared expert, and then 16 experts as a more fine-grained version. All of them underperformed TE V3.

The main warning sign was expert collapse. The router kept sending most tokens to a small number of experts, while the rest were barely used. The metric shown here tracks, on average across training batches, how many experts were used by fewer than 5% of tokens.

We tried several ways to balance expert load, but we quickly ran into a delimma: the model is kinda makes a trade off, either routing became more balanced and task performance got worse, or task performance improved and collapse came back. You can see that tension in the table on the bottom right.

When I added an auxiliary loss, which you can think of as a regularizer as for traditional model, that penalize expert collapse on the objective function, expert collapse improved, but the model still stayed below the dense baseline. A likely explanation is that the auxiliary objective started to dominate the gradients, and give the model lower capacity to optimize the actual prediction task.

I also tried approaches that did not penalize routing through an extra loss term. In those setups, the router gets a small running bias correction: overused experts are nudged down, underused experts are nudged up, and that happens outside the main task gradient. In those cases, performance recovered somewhat, but collapse returned. So the pattern was consistent: the model predicted better when it reused the same small set of experts.

We ran many additional experiments, but the conclusion stayed the same. MoE did not separate the latent concepts in the way we had hoped.

Our working hypothesis is that there are three reasons.

First, there is a scale mismatch. MoE tends to shine at scales much larger than what we are running here. the smallest model size that uses MOE I could find in the literature is around 1B parameters, and the benefits become more clear at 10B+ scale. In contrast, our model is only around 27M parameters, so it may be too small to support multiple experts that can specialize in different regimes. When we doubled the representation size from 256 to 512, the MoE model could finally reach parity with dense TE V3, but it only got there by paying a much larger parameter tax and still showing heavier expert collapse.

Second is the dimension constraints. 256 may be enough for the dense transformer, but still too tight for MoE. if you still remember we have over 75k unique codes as input and condense them to only 256 dimensions to represent them. so with each embedding, there may not sufficient boundaries for the router to find distinct patterns and route them to different experts. We teseted it and it's promising if you look at the final row this is what we tried increasing the embedding dimension to 512, the performance actually improved, but the expert collapse got worse. and the performance is still just on par with the dense baseline but we are taxed with much more parameters. 


Third and and probably more importantly, the data may simply be too homogeneous to support strong expert specialization. Natural language is much more diverse than diagnosis-code sequences. Members absolutely heterogeneous and vary in severity and combinations, but the code patterns themselves repeat much more than we expected.

One of our analyses has proved this. Here, novelty rate means: on a given day in one member timeline, what fraction of that day's codes have not appeared on any earlier day for that same member. So if a given day has 5 codes and only 1 of them has never been seen before in that member history, the novelty rate for that day is 20%.

In the figure, the top-left panel shows that this fraction drops very quickly after the first few days, the top-middle panel shows cumulative unique codes still rising but with a much flatter slope over time, and the remaining panels show that the same decay pattern holds across code frequency tiers, lines of business, age groups, and same-day code pairs.

By day 10, only 24% of codes within a member timeline are still novel. By day 100, that falls to 7%. 

Overall 
the bottleneck here may not be the architecture. It may be the data.