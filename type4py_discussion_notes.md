# Type4Py Discussion
### Date: 04-10-2022

## Q1: What are the main novelties of this paper?

- The model HNN
- Dataset 
- MRR, its a general score for recommendation system, weighted to get better recommendation higher
- TypeCluster explained some more, is unbounded 

## Q2: Why do you think MRR is a good metric for type completion?

- Developers mostly use the first suggestion TOP-1?
- Interesting study idea: look at how important top-2, top-3
- Simple and does the job, but might benefit from more complex number

## Q3: What can we learn from the rise of accuracy from top-1 to top-10? What does that tell us about the model?

- Low performance increase by giving more options
- Most right predictions are predicted quite confidently or not at all
- Also makes sense when you look at the rare column
- MRR is not generally used for these types of tasks

## Q4: What are the limitations of a model that uses type clusters?

- If a cluster is not narrowly defined the range of the cluster can be wide
- not enough data could make small clusters
- limited by the training data
- if types are very similar they can get confused (char vs str), base type match is therefore used
- triplet loss function negates this effect, but you require an example of negative and positive to split these clusters

## Q5: How could you teach a model to synthesize unseen types?

- GNN can model the relationship, heuristic definition of type rules
- Deep nested types with low frequency are hard to predict there cut off nested types
- bootstrapping + static analysis

## Q6: Transformers are currently state-of-the-art, why did the authors use HNN?

- first version released 2020, transformers were relatively new in code application
- practical application, sequence input size could decrease, inference time can be slower with transformers
- identifier sequence is concatenation of all function related names

## Q7: Is there a disadvantage of using HNN over GNN?

- long sequences are better in GNN, but they oversquash their vectors which can results in information loss
- HNN different networks for different embeddings
- number of parameters that need fine-tuning
- HNN more serial, than GNN

## Q8: What are thdictions with limited types for common. e pitfalls of bootstrapping the training data with staticly inferred type annotations and how can we solve this?

- Pyre does not perserve the functionality perfectly, therefore you need type checking and repair
- we need better search strategies, since the training data still lacks type hinting

## Q9: What would you have approached differently?

- Distance metrics might be interested to change and see what happens if changed
- There are many different search strategies that could give other probabilities given the model
- Embeddings became high dimensional and all predictions came really close together. Annoy performs KNN search by approximating.
- Multiple possible and negative for the triplet loss function, might be overfitting the model

## Q10: Our project proposes to add similar type clustering techniques on CodeBERT model. How do you think this affects performance?

- CodeBERT is a multilangual 
- Add a linear layer at the head of the model that maps to a type cluster
- DSL is used



