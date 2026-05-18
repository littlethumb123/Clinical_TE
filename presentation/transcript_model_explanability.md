I am gonna get started by reviewing what i have done with Elle and Jane in last year thinkubator, and then share a framework that's gonna guide our future work on explanability. next I will walk through a few examples how we worked on this before and what we learned about the emebdding and transformer; after that Pritha is gonna share some of interesting findings she has done

When it comes to explanability, we are talking about three things: 1. what does these embedding really mean; 2. which dimensions matter to me, matter to my prediction tasks; 3. how did the transformer develop the understanding of the clinical concepts and convert the raw claims to something that is predictive to business metrics we care about;

we started by asking ourselves these questions and decided to approach this problem using two types of methods; one is task driven, where we care about what the embeddings really mean to our prediction tasks; we use correlation between the embedding dimensions and traditional features to understand what information each dimension is bearing. the 154th dimension may be more aboiut diabetes and 186 dimnension is more about the uitlization patterns. 

we also care about which dimensions really matter to the tasks; This is similar to how we come with feature importance in traditional machine learning models; we look at the shapley values, we look at the impact of model performance if we remove a dimension from teh feature set; we also tried feeding the LLM with members' raw claims and feature importance scores and have it to summarized why a member was considered as the IP risk in the next 3 months. what factors contribute to that risk;

Here gives you a few examples; 
This is we use correlation analysis and find out the 224 dimension is highly predicting the IP risks; and the reason is that it's highly correlated to disease burden; if you look at the heatmap on teh right side; higher dimension values idnicate higher number of chronic conditions; similar thing in the 154 dimension it's indicating diabetes; 
- next slides for LLM
this is the example of using LLM to generate summary based on the raw claims and features importance; explain why a member has high IP risks; 


- GO BACK TO FIRST SLIDE
another aspect is about model behavior where we care about how a transformer convert claims to something useful; 
we were like a surgeon, dissecting the model and looking inside piece by piece; understand how information flows in and out; which part piece captures which clinical concepts, how the mode construct the understanding of a concept.. I know this is very confusing and abstractive, I will give you an example 

The transformer consists of a series of layers; you can think about each layer is a machine, they are positioned in a sequence, they process the information one by one from the beginning to the end; 
the left chart shows how well each machine can understand the clinical concepts, like the anxiety, diabetes, hypertension and social risks; we can see the trajecotry is going up and that means the knowledge of each concepts is accumulated as it goes through the layers; how I did this is something called probing, I took each machine out of the model and have it to predict the concept of interests among mmeber popualtions. the y-axis indicate the auc-roc of such prediction 

ok now we know that the transformer is gradually picking up the knowlegde of the concepts; but do all the machines learn the same thing or they learn different things? 
If you look at the right chart, it shows the homogeneity of the information that flows across different machines; at the very beginning, the model thought all clinical concepts are the same; they are the same information; so you can see the similarity is very high at the very beginning; and as the informaiton going forwad, the following machines started to learn that these concepts are not the same; they are different; the hypertension should be more similar to diabetes than to social risk. this is exactly what we see in the right part of the chart; the model is able to understand the difference between these concepts; 
This tells us that the machines positioned at front are learning some knowledge more generic and common, whereas the later machines are learning more clinical concept specific; 

How I did this is again I take the machine out of the model and have it to predict the concept of the interests using a logistic regression; now mathmatcally, mathmatcially in high dimensional feature space, we could use this vector to represent that concept; then I calculate the similarity between concepts using their corresponding vectors; 

This lesson opens up some opportunities to further understand the mechanism of the concept learning in the model and how to improve this capability. and also we can use the same method to analyze which dimension is correlated to which concepts; Pritha is gonna share more details

After these thinkubator works we can come up with a more structural and informative framework to tie all these together; show what we expect the explanability to be: 
From a user perspective, we expect the explanation to tell me what the embedding really means, what matters to my task and how it builds up the nderstanding; 
It should also be inclusive to different users; right now we are pimarily facing data scientists team but we shoudl also keep in mind that the CMs are the end users of the data scientist model; what we explain to DS should be also making sense to CM and able to help DS to explain to CMs. 
In order to do that; we have four principals we believe important to follow: 
First faithfullness, the explanations should reflect what the model actually does, not just what sounds reasonable. this is why we understand model behaviors, not only the task oriented explanations; 
Second, consistency — the explanations should keep the same and stable given the same input and context; it shouldn't be that the dimension A means diabetes for medicaid while means hypertension to medicare;
third, transparency, the transparency is operated at three levels, the method we used to explain should be transparent, the procedure we conduct the explanation should be transparent and the limitations or what we coundn't explain should be transparent to users; this is why we should be cautious about hte LLM-based explanations; it is very promising and in Jane's project it shows some interesting results; but we should make sure we understand what we are feeding the LLM and why we think it's useful to feed such information for better explanations; otherwise we are just asking a black box to explain a black box. 
lastly, completeness; the explanations should capture a sufficient portion of the outputs and model mechanisms. for example we should have clear understanding over 50% of the dimensions and thier impacts on the IP or RAP models; how many concepts it can capture well. 

For the methods there are tons of them and I am not going through this. Pritha.. .








