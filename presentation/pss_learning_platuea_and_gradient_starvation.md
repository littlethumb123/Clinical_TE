### Milestones and Timeline

Let me start with where we are and how we got here.

So back in mid-October, we started the architecture research and data prep for retraining. We tried a lot of different designs—actually 15 different architectures—and by end of November we narrowed it down to three that looked really promising. We trained all three, along with the legacy Transformer, on about 1.7 million members. Then we presented those results to Eric and Aillson's team in CAPS meeting.

Now we have moved into the downstream evaluation. Basically, we're taking the embeddings from these new transformers and testing them against the business models: can they actually predict inpatient risk for Medicare and Commercial? 

The reasoning being is that —full training is expensive. We're talking months of work and significant GPU costs. We also can't exactly reproduce the original training setup from a few years back. So if there's a problem, we want to know now, instead after we've spent all that time and compute resources. This checkpoint gives us a chance to validate early.

Right now we're digging into the results. Trying to ask if the embeddings are actually predictive? Did it learn what we wanted it to learn? What's working, what's not, and why? All of this will guide us to the next formal training. 

We're targeting March to April to kick off the full-scale evaluation with Yiwei's team. Once we get sign-off there, we're aiming for production around end of Q2, maybe early Q3.

---

### Results

Alright, let's look at what we found.

So this table shows how well the IP models perform with the use of embeddings —for both Medicare and Commercial. We're comparing three types of features: traditional tabular features that production IP model is using, embedding-only features from the Transformer, and a hybrid that uses both. What we did is that we replicated the IP model training pipelines; if the production model uses xgboost, we use xgboost, if it uses catboost, we used catboost; and we trained and evaluate the pipeline with those three types of features. 

The columns across the top indicates different Transformer types where the embedding comes from. On the left is the original legacy TE. Then we have our three new ones: one with an optimized training strategy, one that adds Flash Attention, and one with Mixture of Experts. These variants are added incrementally to see if they can outperform the legacy TE.
each row indicates the performance metrics

One thing we can clearly see is that all three new architectures perform better than the legacy TE. That's what we expected; 

Then For Medicare, the embedding-only models perform almost identically to feature engineered model. the hybrid is slightly better. This is also what we expect; It means embeddings can potentially replace the manual feature engineering we do today.

For Commercial, it's a similar, just a bit weaker. The new TEs come in slightly below engineered features, but the hybrid basically matches it.

And one thing to note is that — we only trained on 1.7 million members here. When we scale up to 16 million, we would expect to see these numbers to get even better.

So overall the signal is positive. We're on the right track.

---

### Deep Dive 1: Embedding Quality

But this is not the whole story; the training performance is just one aspect; We wanted to really understand—what did the Transformer actually learn? Are these embeddings capturing meaningful clinical information?

So we did a deep dive. We looked at all 256 embedding dimensions and visualized how well they can separate IP members from non-IP members. Kudos to Pritha for doing this very nice clustering analysis 

On the left hand side is a cluster plot. Each dot is a member, positioned based on their embedding. What you can see is that non-IP members tend to cluster toward the top-left, while IP members pull toward the other side. There's no clean line between them, but we can definitely see the separation. That means that the embeddings carry real signal for identifying the members at IP risk. 

and another question brought up, how many of the embedding dimensions are actually useful? On the right, we're looking at correlations. Each bar shows how strongly one embedding dimension correlates with IP outcome. 
about 46% of dimensions have correlation above 0.05. So roughly half our embedding space is picking up something relevant to IP. the guy on the far right with teh highest correlation encoded multifaceted information about high claim volume and ER utilizations; these are actually the important features identified in the production IP model; so it's encouraging to see that the Transformer captures the right things.

OK now we have seen that embeddings can match traditional features—especially for Medicare. We see good separation in the clusters. Half the dimensions show relevance. But beyond this—we expected embeddings to *outperform* the engineered features. So why didn't that happen? Or let me put it another way: did the Transformer actually learn what we wanted it to learn?
---

### Deep Dive 2: The Learning Plateau

So we did another deep dive; 
Quick reminder on how transformer works. The Transformer is trained to predict what claim codes a member will have tomorrow. The embeddings are just a byproduct of that task. So we went back and looked at how the model was learning during training.

if you take a look at the left graph. This shows the training loss and recall metrics over time. When we're training a neural network, we want that loss to keep going down—that means the predictions are getting better and better.

if you look at the red lines, Loss drops really fast at first, then just... flattens out. Same thing with the blue and green lines—those are Recall@10 and Micro Recall@10. They improve quickly, then plateau around step 8,000. The model just stopped getting better.

So why did it stop?

We did another analysis—this time we tracked the gradient flow. Think of gradients like nutrients for the model. They're the signal that tells the model how to learn how to improve. If that signal gets too weak, learning is gonna stop.

Now look at the right graph. This shows how those gradients are distributed across different types of codes codes with different frequencies—common ones, medium frequency, rare, and very rare (what we call tail).

At the beginning, things are pretty balanced. Each tier gets a fair share of the learning signal. But as training goes on, something happens. The common codes start taking over. They grab more and more of the gradient and all the way to the end they were consuming 85% of the learning signal. The medium, rare and tail codes are basically getting almost nothing and even died out.

This is what we call gradient starvation.

What's happening is the model learns the common stuff fast—because it sees it all the time. But the rare codes are harder to learn, they show up less often, and eventually the model just... gives up on learning.

These two graphs give us a means that our embeddings get really good at representing common conditions, but are weak on rare ones. This limits the representative capability of our embeddings and partially explains why performance plateaued.

### What we learnt
The downstream evaluation is encouraging that the TE embeddings captures relevant signals to the IP and potentially matches human-feature engineered model. more importantly through these indepth post training analysis we gain a capability to explain why TE works why it's not working and what the embeddings really means clinically.

Now teh question still remains, why embedding did not outperform the tabular features? we have confirmed one hypothesis; and other reasons might be the data. Also we are training 

Next we 













---

### Technical Summary

**Learning Plateau:**
Loss drops rapidly (0.80 → 0.005 in ~1,000 steps), then stabilizes around 0.003. Recall@10 saturates at ~80%, Micro Recall@10 at ~58%. The model hits a local optimum with diminishing returns from further training.

**Gradient Starvation:**
Common tier gradient share grows from 18% → 85%. Tail tier collapses from 18% → 0.1%. Frequent codes monopolize updates; rare codes receive almost no learning signal.

---

### Plain Language Summary

**The Plateau:**
The model learns fast at first, then just stops improving. Like a student who picks up the easy stuff quickly, then hits a wall. No matter how much longer you train, it doesn't get better.

**Gradient Starvation:**
Common diagnoses hog all the learning. Imagine a classroom where the teacher only calls on the loud kids—the quiet ones never get heard. Common conditions end up taking 85% of what the model learns. Rare but important conditions? They get almost nothing. So the model gets great at predicting the usual stuff, but struggles with the rare cases—which are often the ones that matter most.
