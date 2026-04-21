### VAE latent space vs transformer representations
#### Variational Auto-Encoder Latent Space
- A VAE learns from a compressed, continuous bottleneck(latent vector `z`) that attempts to summarize the input in a way that supports *sampling and smooth interpolation(similar z -> similar decoded outputs)*
- Tricks like reconstruction plus regularization allows for continuous and stochastic interpretation of input(Gaussian Randomness) that are clustered in a low-dimensional structure(close together so meaningful connection between clusters)
#### Transformer representations
- A transformer builds contextual embeddings: each position's vector depends on *the whole sequence(via attention), not only on local reconstruction of tokens(latent space)*
- No fixed bottleneck latent representation, there are layers of learned contextualization that offer richer and task-adapted representations
- Geometry is not so much about a smooth generative manifold and more about the degree of linear separability of concepts- What information is linearly readable at each layer

### Reconstruction objective vs masked language modeling

#### Reconstruction (VAE - style)
- The model must copy/predict the input through a bottleneck(manifold)
- *Pressure*: preserve everything needed to rebuild the input—often low-level details unless the bottleneck is tight.
- *Risk*: the model may focus on pixel/token fidelity rather than abstract semantics, unless architecture or constraints push abstraction.
- This pressure may prioritize the fidelity of individual tokens rather than abstract semantics
#### Masked language modeling(MLM)
- The model sees partial context and must fill in missing tokens(over many iterations and sub-iterations)
- *Pressure*: use bidirectional(which earlier/later tokens influence later/earlier tokens) context to infer likely tokens—strongly aligned with language/statistics of the domain and often semantic and syntactic regularities.
- It is not “copy the whole input”; it is predict what fits, which can encourage rich contextual understanding in hidden states.

Summary:
	Reconstruction prioritizes informations preservation under compression in an attempt to extrapolate learned relationships between states
	On the other hand, MLMs are equivalent to conditional prediction from context. They build internal features that summarize what the sequence is about and how it is formed.
	The model benefits from representing high-level properties(topic, grammatical roles, domain context in proteins)
	Those same high-level properties what downstream tasks need, hidden states are a bank for recognizing features that may not be recognizable

### Why probing helps interpret representation quality
#### What is probing?
 - Train a simple predictor(linear or small MLP) on top of frozen representations(maybe a slice of our transformer somewhere(hidden layer)) to predict some external label you care about(properties, structure, class, etc.)
 - Pick a particular layer n's hidden states $h(^n)$, freeze the transformer, and train a small linear probe to predict label y from $h(^n)$
#### This tells us...
- High probe accuracy(w/ a linear one): the info needed for predicting label y is present and easily readable in layer n
	- The input data is sufficiently linear separable to indicate strong predictive accuracy
- Low probe accuracy: the info is absent or its encoded in a harder-to-read way(the input data is convoluted and non deterministic, non-linearly separable)

#### Useful?
- The transformer is less a black box and more of a **layer-wise story***
- Its possible to see where properties emerge and its characteristic that early layers predict local patterns, and later more task/objective specific general trending patterns

Summary: Probing shows decodability not usage. Is concept y recoverable and from layer n with a simple readout, not necessarily whether layer n is good at the models global task


