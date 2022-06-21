# Natural-Language-Processing
The University of Melbourne - Natural Language Processing (COMP90042)

This respiratory cantains Final exam paper of Natural Language Processing (COMP90042)
The Assignment 1 and Assignment 2

Notes:

Natural Language Processing (COMP90042)

Lecture 1

感觉这一节课好像没有什么内容。


Lecture 2 (Text Preprocessing)
Discrete representation
局限：不能捕捉语义关系
向量长稀疏

Why Preprocess?
language is compositional.
Preprocessing is the first step.

Preprocessing Steps
Remove unwanted formatting (e.g. HTML)
Sentence segmentation(分割): break documents into sentences
Word tokenisation: break sentences into words
Word normalisation: transform words into canonical(典范) forms
Stopword removal: delete unwanted words

Sentence Segmentation
Naïve approach: break on sentence punctuation ([.?!])

Second try: use regex to require capital ([.?!] [A-Z])

Better yet: have lexicons
·But difficult to enumerate all names and abbreviations

Binary Classifier
Look at every “.” and decide whether it is the end of a sentence.
Decision trees, logistic regression
Features
‣ Look at the words before and after “.”
‣ Word shapes:
- Uppercase, lowercase, ALL_CAPS, number
- Character length
‣ Part-of-speech tags:
- Determiners tend to start a sentence


Word Tokenisation
Word Tokenisation: English
Naïve approach: separate out alphabetic strings (\w+)

Word Tokenisation: Chinese
Some Asian languages are written without spaces between words

In Chinese, words often correspond to more than one character

Standard approach assumes an existing vocabulary

MaxMatch algorithm
Greedily match longest word in the vocabulary

But how do we know what the vocabulary is
And doesn’t always work

Word Tokenisation: German
Requires compound(复合物) splitter


Subword Tokenisation
One popular algorithm: byte-pair encoding (BPE)
Core idea: iteratively merge frequent pairs of characters
Advantage:
‣ Data-informed tokenisation
‣ Works for different languages
‣ Deals better with unknown words

Some features:

In practice BPE will run with thousands of merges, creating a large vocabulary

Most frequent words will be represented as full words

Rarer words will be broken into subwords

In the worst case, unknown words in test data will be broken into individual letter

Disadvantages:
Creates non-sensical subwords

Word Normalisation
Lower casing (Australia → australia)
Removing morphology (cooking → cook)
Correcting spelling (definately → definitely)
Expanding abbreviations (U.S.A → USA)
Goal:
	‣ Reduce vocabulary
	‣ Maps words into the same type

Inflectional Morphology(屈折形态)
Inflectional morphology creates grammatical variants
English inflects nouns, verbs, and adjectives
	‣ Nouns: number of the noun (-s)
	‣ Verbs: number of the subject (-s), the aspect (-ing) of the action and the tense (-ed) of the action
‣ Adjectives: comparatives (-er) and superlatives (-est)
Many languages have much richer inflectional morphology than English

Lemmatisation (词形还原)
Lemmatisation means removing any inflection to reach the uninflected form, the lemma
	speaking → speak
In English, there are irregularities that prevent a trivial solution:
‣ poked → poke (not pok)
‣ stopping → stop (not stopp)
	‣ watches → watch (not watche)
‣ was → be (not wa)

A lexicon(词典) of lemmas needed for accurate lemmatisation

Derivational Morphology(衍生形态)
Derivational morphology creates distinct words

English derivational suffixes(后缀) often change the lexical category, e.g.
‣ -ly (personal → personally)
‣ -ise (final → finalise)
‣ -er (write → writer)

English derivational prefixes(前缀) often change the meaning without changing the lexical category
‣ write → rewrite
‣ healthy → unhealthy

Stemming(词干)
Stemming strips off all suffixes, leaving a stem
·E.g. automate, automatic, automation → automat
·Often not an actual lexical item

Even less lexical sparsity than lemmatisation

Popular in information retrieval

Stem not always interpretable

The Porter Stemmer
Most popular stemmer for English
Applies rewrite rules in stages
First strip inflectional suffixes,
		- E.g. -ies → -i
	Then derivational suffixes
		- E.g -isation → -ise → -i
Stopword Removal

Definition of stopword: a list of words to be removed from the document
‣ Typical in bag-of-word (BOW) representations
	‣ Not appropriate when sequence is important



Lecture 3 (N-gram Language Models)
Language Models
One NLP application is about explaining language
‣ Why some sentences are more fluent than others

E.g. in speech recognition:
	recognise speech > wreck a nice beach

We measure ‘goodness’ using probabilities estimated by language models

Language model can also be used for generation

Probabilities: Joint to Conditional
Our goal is to get a probability for an arbitrary sequence of m words

P(w1,w2, … ,wm)

First step is to apply the chain rule to convert joint probabilities to conditional ones
P(w1,w2, . . . ,wm) = P(w1)P(w2 |w1)P(w3 |w1,w2) … P(wm |w1, . . . ,wm−1)


The Markov Assumption
Still intractable, so make a simplifying assumption:
 

For some small n
When n = 1, a unigram model
 
When n = 2, a bigram model
 

When n = 3, a trigram model
 


Maximum Likelihood Estimation
How do we calculate the probabilities? Estimate based on counts in our corpus:

For unigram models,
 

For bigram models,
 

For n-gram models generally,
 
Book-ending Sequences

Special tags used to denote start and end of sequence
<s> = sentence start
</s> = sentence end

Trigram example
具体的计算细节就不在这里写了，可以看一下PPT

Several Problems
Language has long distance effects — need large n
‣ The lecture/s that took place last week was/were on preprocessing.

Resulting probabilities are often very small
‣ Use log probability to avoid numerical underflow

What about unseen n-grams?
‣ Need to smooth the LM!

Smoothing
Basic idea: give events you’ve never seen before some probability

Must be the case that P(everything) = 1

Laplacian (Add-one) Smoothing

Simple idea: pretend we’ve seen each n-gram once more than we did.

For unigram models (V= the vocabulary)
 

For bigram models,

 

Add-one Example

<s> the rat ate the cheese </s>

What’s the bigram probability P(ate | rat) under add-one smoothing?

 

What’s the bigram probability P(ate | cheese) under add-one smoothing?
 

Add-k Smoothing

Adding one is often too much
Instead, add a fraction k
AKA Lidstone Smoothing, or Add- Smoothing

 
Have to choose k


Lidstone Smoothing
 

Absolute Discounting
‘Borrows’ a fixed probability mass from observed n-gram counts

Redistributes it to unseen n-grams
 


Backoff
Absolute discounting redistributes the probability mass equally for all unseen n-grams
Katz Backoff: redistributes the mass based on a lower order model (e.g. unigram)
 

Issues with Katz Backoff
I can’t see without my reading ___
C(reading, glasses) = C(reading, Francisco) = 0
C(Francisco) > C(glasses)
Katz backoff will give higher probability to Francisco

Kneser-Ney Smoothing
Redistribute probability mass based on the versatility(多功能性) of the lower order n-gram
AKA “continuation probability”
What is versatility?
	High versatility -> co-occurs with a lot of unique words, e.g. glasses
		- men’s glasses, black glasses, buy glasses, etc
Low versatility -> co-occurs with few unique words, e.g. francisco
		- san francisco
 

Intuitively the numerator of Pcont counts the number of unique wi-1 that co-occurs with wi

High continuation counts for glasses
Low continuation counts for Franciso

Interpolation
A better way to combine different orders of n-gram models

Interpolated trigram model:

 

Lecture 4 (Text Classification)
Classification
Input
A document d
Often represented as a vector of features
A fixed output set of classes C = {c1,c2,...ck}
	A fixed output set of classes C = {c1,c2,...ck}
		Categorical, not continuous (regression) or ordinal (ranking)


Output
A predicted class c ∈ C

Text Classification Tasks
Some common examples
	‣ Topic classification
‣ Sentiment analysis
‣ Native-language identification
‣ Natural language inference
‣ Automatic fact-checking
‣ Paraphrase

Input may not be a long document
	E.g. sentence or tweet-level sentiment analysis

Topic Classification
Motivation: library science, information retrieval
Classes: Topic categories, e.g. “jobs”, “international news”

Sentiment Analysis
Motivation: opinion mining, business analytics

Classes: Positive/Negative/(Neutral)

Features
	‣ N-grams
‣ Polarity lexicons(反义词典，positive and negative dictonary)

Native-Language Identification
Motivation: forensic linguistics, educational applications
Classes: first language of author (e.g. Indonesian)

Features
‣ Word N-grams
‣ Syntactic patterns (POS, parse trees)
‣ Phonological features

Examples of corpora
‣ TOEFL/IELTS essay corpora

Natural Language Inference
AKA textual entailment
Motivation: language understanding
Classes: entailment, contradiction, neutral
Features
	‣ Word overlap
‣ Length difference between the sentences
‣ N-grams

Building a Text Classifier

1. Identify a task of interest
2. Collect an appropriate corpus
3. Carry out annotation
4. Select features
5. Choose a machine learning algorithm
6. Train model and tune hyper-parameters using held-out development data
7. Repeat earlier steps as needed
8. Train final model
9. Evaluate model on held-out test data


Algorithms for Classification
Choosing a Classification Algorithm
Bias vs. Variance
	Bias: assumptions we made in our model
Variance: sensitivity to training set

Underlying assumptions, e.g., independence
Complexity
Speed

Naïve Bayes
Finds the class with the highest likelihood under Bayes law

Naïvely assumes features are independent

Advantage:
Fast to train and classify
robust, low-variance → good for low data situations
optimal classifier if independence assumption is correct
extremely simple to implement.

Disadvantage:
Independence assumption rarely holds
low accuracy compared to similar methods in most situations
smoothing required for unseen class/feature combinations

Logistic Regression
A classifier, despite its name
A linear model, but uses softmax “squashing” to get valid probability
Training maximises probability of training data subject to regularisation which encourages low or sparse weights

Pros:
Unlike Naïve Bayes not confounded by diverse, correlated features → better performance

Cons:
Slow to train
Feature scaling needed
Requires a lot of data to work well in practice
Choosing regularisation strategy is important since overfitting is a big problem

Support Vector Machines
Finds hyperplane which separates the training data with maximum margin

Pros:
Fast and accurate linear classifier
Can do non-linearity with kernel trick
Works well with huge feature sets

Cons:
Multiclass classification awkward
Feature scaling needed
Deals poorly with class imbalances
Interpretability

K-Nearest Neighbour
Classify based on majority class of k-nearest training examples in feature space

Definition of nearest can vary
	Euclidean distance
	Cosine distance

Pros:
Simple but surprisingly effective
No training required
Inherently multiclass
Optimal classifier with infinite data

Cons:
Have to select k
Issues with imbalanced classes
Often slow (for finding the neighbours)
Features must be selected carefully

Decision tree
Construct a tree where nodes correspond to tests on individual features

Leaves are final class decisions

Based on greedy maximisation of mutual information

Pros:
Fast to build and test
Feature scaling irrelevant
Good for small feature sets
Handles non-linearly-separable problems

Cons:
In practice, not that interpretable
Highly redundant sub-trees
Not competitive for large feature sets

Random Forests
An ensemble(合奏) classifier

Consists of decision trees trained on different subsets of the training and feature space

Final class decision is majority vote of sub-classifiers

Pros:
Usually more accurate and more robust than decision trees
Great classifier for medium feature sets
Training easily parallelised

Cons:
Interpretability
Slow with large feature sets

Neural Networks
An interconnected set of nodes typically arranged in layers

Input layer (features), output layer (class probabilities), and one or more hidden layers

Each node performs a linear weighting of its inputs from previous layer, passes result through activation function to nodes in next layer

Pros:
Extremely powerful, dominant method in NLP and vision
Little feature engineering

Cons:
Not an off-the-shelf classifier
Many hyper-parameters, difficult to optimise
Slow to train
Prone to overfitting

Hyper-parameter Tuning
Dataset for tuning
Development set
Not the training set or the test set
k-fold cross-validation
Specific hyper-parameters are classifier specific
	E.g. tree depth for decision trees
But many hyper-parameters relate to regularisation
	Regularisation hyper-parameters penalise model complexity
	Used to prevent overfitting

For multiple hyper-parameters, use grid search

Evaluation
Accuracy
Accuracy = correct classifications/total classifications

Precision & Recall
Precision = correct classifications of B (tp)/ total classifications as B (tp + fp)

Recall = correct classifications of B (tp)/ total instances of B (tp + fn)

F(1)-score (F1)
Harmonic(谐波; 和声的) mean of precision and recall

 
Like precision and recall, defined relative to a specific positive class

But can be used as a general multiclass metric
Macroaverage: Average F-score across classes
Microaverage: Calculate F-score using sum of counts (= accuracy for multiclass problems)

Lecture 5 (Part of Speech Tagging)
What is Part of Speech (POS)?
AKA word classes, morphological(形态学) classes, syntactic categories
Nouns, verbs, adjective, etc
POS tells us quite a bit about a word and its neighbours:

‣ nouns are often preceded by determiners
‣ verbs preceded by nouns
‣ content as a noun pronounced as CONtent
‣ content as a adjective pronounced as conTENT


Information Extraction
Given this:
‣ “Brasilia, the Brazilian capital, was founded in 1960.”


Obtain this:
‣ capital(Brazil, Brasilia)
‣ founded(Brasilia, 1960)

Many steps involved but first need to know nouns (Brasilia, capital), adjectives (Brazilian), verbs (founded) and numbers (1960).

POS Open Classes

Open vs closed classes: how readily do POS categories take on new words? Just a few open classes:
Nouns
Proper (Australia) versus common (wombat)
Mass (rice) versus count (bowls)

Verbs
	Rich inflection (go/goes/going/gone/went)
	Auxiliary verbs (be, have, and do in English)
	Transitivity (wait versus hit versus give)
— number of arguments

Ambiguity
Many word types belong to multiple classes
POS depends on context

Compare:

‣ Time flies like an arrow
‣ Fruit flies like a banana

 

POS Ambiguity in News Headlines


Tagsets
A compact representation of POS information
Usually ≤ 4 capitalized characters (e.g. NN = noun)
Often includes inflectional distinctions

Major English tagsets
Brown (87 tags)
Penn Treebank (45 tags)
CLAWS/BNC (61 tags)
	“Universal” (12 tags)

At least one tagset for all major languages

Major Penn Treebank Tags
NN noun VB verb

JJ adjective
RB adverb
DT determiner
CD cardinal number
IN preposition
PRP personal pronoun
MD modal
CC coordinating conjunction
RP particle
WH wh-pronoun
TO to

Derived Tags (Open Class)

• NN (noun singular, wombat)

‣ NNS (plural, wombats)
‣ NNP (proper, Australia)
‣ NNPS (proper plural, Australians)

• VB (verb infinitive, eat)

‣ VBP (1st /2nd person present, eat)
‣ VBZ (3rd person singular, eats)
‣ VBD (past tense, ate)
‣ VBG (gerund, eating)
‣ VBN (past participle, eaten)

• JJ (adjective, nice)

‣ JJR (comparative, nicer)
‣ JJS (superlative, nicest)

• RB (adverb, fast)

‣ RBR (comparative, faster)
‣ RBS (superlative, fastest)

• PRP (pronoun personal, I)

‣ PRP$ (possessive, my)

• WP (Wh-pronoun, what):

‣ WP$ (possessive, whose)
‣ WDT(wh-determiner, which)
‣ WRB (wh-adverb, where)



Page 20:
 
Automatic Tagging

Why Automatically POS tag?
Important for morphological(形态学) analysis, e.g. lemmatisation

For some applications, we want to focus on certain POS
E.g. nouns are important for information retrieval, adjectives for sentiment analysis

Very useful features for certain classification tasks
E.g. genre attribution (fiction vs. non-fiction)

POS tags can offer word sense disambiguation
	E.g. cross/NN vs cross/VB cross/JJ

Automatic Taggers

Rule-based taggers

Statistical taggers
Unigram tagger
Classifier-based taggers
Hidden Markov Model (HMM) taggers

Rule-based tagging
Typically starts with a list of possible tags for each word
From a lexical resource, or a corpus

Often includes other lexical information, e.g. verb subcategorisation (its arguments)

Apply rules to narrow down to a single tag
E.g. If DT comes before word, then eliminate VB
Relies on some unambiguous contexts

Large systems have 1000s of constraints

Unigram tagger
Assign most common tag to each word type
Requires a corpus of tagged words
“Model” is just a look-up table
But actually quite good, ~90% accuracy
Correctly resolves about 75% of ambiguity

Often considered the baseline for more complex approaches

Classifier-Based Tagging
Use a standard discriminative classifier (e.g. logistic regression, neural network), with features:
Target word
Lexical context around the word
Already classified tags in sentence

But can suffer from error propagation: wrong predictions from previous steps affect the next ones

Hidden Markov Models
A basic sequential (or structured) model

Like sequential classifiers, use both previous tag and lexical evidence

Unlike classifiers, considers all possibilities of previous tag

Unlike classifiers, treat previous tag evidence and lexical evidence as independent from each other
Less sparsity
Fast algorithms for sequential prediction, i.e. finding the best tagging of entire word sequence

Unknown Words
Huge problem in morphologically rich languages (e.g. Turkish)

Can use things we’ve seen only once (hapax legomena) to best guess for things we’ve never seen before
	Tend to be nouns, followed by verbs
	Unlikely to be determiners
Can use sub-word representations to capture morphology (look for common affixes)

Lecture 6 (Sequence Tagging: Hidden Markov Models)
A Better Approach
Tagging is a sentence-level task but as humans we decompose it into small word-level tasks.
Solution:
Define a model that decomposes process into individual word level steps
But that takes into account the whole sequence when learning and predicting (no error propagation)

This is the idea of sequence labelling, and more general, structured prediction.

A Probabilistic Model
Goal: obtain best tag sequence t from sentence w
 
Let’s decompose:
 

This is a Hidden Markov Model (HMM)

Two Assumptions of HMM
 

HMMs - Training
Parameters are the individual probabilities

 

Training uses Maximum Likelihood Estimation (MLE)
This is done by simply counting word frequencies according to their tags (just like N-gram LMs!)

 

What about the first tag?

Assume we have a symbol “<s>” that represents the start of your sentence

 

What about unseen (word, tag) and (tag, previous_tag) combinations?
	Smoothing techniques


Transition Matrix
 

Emission (Observation) Matrix


HMMs – Prediction (Decoding)
 

Simple idea: for each word, take the tag that maximises. Do it left-to-right, in greedy fashion.


Correct way: consider all possible tag combinations, evaluate them, take the max


The Viterbi Algorithm
Dynamic Programming to the rescue!
	We can still proceed sequentially, as long as we are careful.

Instead, we keep track of scores for each tag for “can” and check them with the different tags of “play”.

Example

Complexity: O(T2N), where T is the size of the tagset and N is the length of the sequence.
T * N matrix, each cell performs T operations.

Why does it work?
‣ Because of the independence assumptions that decompose the problem.
‣ Without these, we cannot apply DP.

Viterbi Pseudocode
• Good practice: work with log probabilities to prevent underflow (multiplications become sums)
• Vectorisation (use matrix-vector operations)


HMMs In Practice
 

We saw HMM taggers based on bigrams (first order HMM).
‣ I.e. current tag depends only the immediate previous tag

State-of-the-art use tag trigrams (second order HMM).

Need to deal with sparsity: some tag trigram sequences might not be present in training data

With additional features, reach 96.5% accuracy on Penn Treebank (Brants, 2000)

Generative vs. Discriminative Taggers
HMM is generative
 

trained HMM can generate data (sentences)!

allows for unsupervised HMMs: learn model without any tagged data!

Discriminative models describe P(T|W) directly

supports richer feature set, generally better accuracy when trained over large supervised datasets

E.g., Maximum Entropy Markov Model (MEMM), Conditional random field (CRF)

Most deep learning models of sequences are discriminative

HMMs in NLP
 
HMMs are highly effective for part-of-speech tagging
-trigram HMM gets 96.5% accuracy
-related models are state of the art
 ‣ MEMMs 97%
 ‣ CRFs 97.6%
 ‣ Deep CRF 97.9%

Apply out-of-the box to other sequence labelling tasks
‣ named entity recognition (lecture 18), shallow parsing, ...
‣ In other fields: DNA, protein sequences...


Lecture 7 (Deep Learning for NLP: Feedforward Networks)
Deep Learning
A branch of machine learning

Re-branded name for neural networks

Why deep? Many layers are chained together in modern deep learning models

Neural networks: historically inspired by the way computation works in the brain
Consists of computation units called neurons

Feed-forward NN
Aka multilayer perceptrons

Each arrow carries a weight, reflecting its importance

Certain layers have non-linear activation functions

Neuron
 

Each neuron is a function
given input x, computes real-value (scalar) h
scales input (with weights, w) and adds offset (bias,b)
applies non-linear function, such as logistic sigmoid, hyperbolic sigmoid (tanh), or rectified linear unit
w and b are parameters of the model

Matrix Vector Notation
 

Typically have several hidden units, i.e.
Each with its own weights ( ) and bias term ( )
Can be expressed using matrix and vector operators
h = tanh (W x + b)
Where is a matrix comprising the weight vectors, and is a vector of all bias terms
Non-linear function applied element-wise

Output Layer
Binary classification problem
	e.g. classify whether a tweet is + or - in sentiment
sigmoid activation function

Multi-class classification problem
	e.g. native language identification
	softmax ensures probabilities > 0 and sum to 1

 

Learning from Data

How to learn the parameters from data?
Consider how well the model “fits” the training data, in terms of the probability it assigns to the correct output
 
want to maximise total probability, L
equivalently minimise -log L with respect to parameters

Trained using gradient descent
	tools like tensorflow, pytorch, dynet use autodiff to compute gradients automatically

Regularisation
Have many parameters, overfits easily

Low bias, high variance

Regularisation is very very important in NNs

L1-norm: sum of absolute values of all parameters (W, b, etc)

L2-norm: sum of squares

Dropout: randomly zero-out some neurons of a layer

Dropout
If dropout rate = 0.1, a random 10% of neurons now have 0 values
Can apply dropout to any layer, but in practice, mostly to the hidden layers

Why Does Dropout Work?
It prevents the model from being over-reliant on certain neurons

It penalises large parameter weights

It introduces noise into the network

Applications in NLP
Topic Classification
Given a document, classify it into a predefined set of topics (e.g. economy, politics, sports)
Input: bag-of-words

Topic Classification - Training
 
Topic Classification - Prediction
 

Topic Classification - Improvements
+ Bag of bigrams as input

Preprocess text to lemmatise(词形还原) words and remove stopwords

Instead of raw counts, we can weight words using TF-IDF or indicators (0 or 1 depending on presence of words)

Language Model Revisited
Assign a probability to a sequence of words

Framed as “sliding a window” over the sentence, predicting each word from finite context E.g., n = 3, a trigram model
 

Training involves collecting frequency counts
	Difficulty with rare events → smoothing

Language Models as Classifiers
LMs can be considered simple classifiers, e.g. for a trigram model:
 
classifies the likely next word in a sequence, given “salt” and “and”.

Feed-forward NN Language Model
Use neural network as a classifier to model

Input features = the previous two words

Output class = the next word

How to represent words? Embeddings
 

Word Embeddings
Maps discrete word symbols to continuous vectors in a relatively low dimensional space

Word embeddings allow the model to capture similarity between words
‣ dog vs. cat
‣ walking vs. running

Topic Classification
 

Training a FFNN LM
 

 

Input and Output Word Embeddings
 
Language Model: Architecture

 

Mostly are softmax computations

Advantages of FFNN LM
Count-based N-gram models (lecture 3)
‣ cheap to train (just collect counts)
‣ problems with sparsity and scaling to larger contexts
‣ don’t adequately capture properties of words (grammatical and semantic similarity), e.g., film vs movie

FFNN N-gram models
‣ automatically capture word properties, leading to more robust estimates


What Are The Limitations of Feedforward NN Language Model?
Very slow to train
Captures only limited context
Unable to handle unseen words

Convolutional Networks
Commonly used in computer vision
Identify indicative local predictors
Combine them to produce a fixed-size representation

Convolutional Networks for NLP
Sliding window (e.g. 3 words) over sequence

W = convolution filter (linear transformation+tanh)

max-pool to produce a fixed-size representation

Final Words
• Pros

‣ Excellent performance
‣ Less hand-engineering of features
‣ Flexible — customised architecture for different tasks

• Cons

‣ Much slower than classical ML models... needs GPU
‣ Lots of parameters due to vocabulary size
‣ Data hungry, not so good on tiny data sets
‣ Pre-training on big corpora helps


Lecture 8 (Deep Learning for NLP:Recurrent Networks)
N-gram Language Models
Can be implemented using counts (with smoothing)
Can be implemented using feed-forward neural networks
Generates sentences like (trigram model):
‣ I saw a table is round and about

Problem: limited context

Recurrent Neural Networks (RNN)
Allow representation of arbitrarily(任意) sized inputs

Core Idea: processes the input sequence one at a time, by applying a recurrence formula

Uses a state vector to represent contexts that have been previously processed

 

 

RNN Unrolled
 

RNN Training

An unrolled RNN is just a very deep neural network

But parameters are shared across all time steps

To train RNN, we just need to create the unrolled computation graph given an input sequence

And use backpropagation algorithm to compute gradients as usual

This procedure is called backpropagation through time


(Simple) RNN for Language Model

 


What Are Some Potential Problems with This Generation Approach?
Mismatch between training and decoding
Error propagation: unable to recover from errors in intermediate steps
Tends to generate “bland” or “generic” language

Long Short-term Memory Networks

Language Model... Solved?
RNN has the capability to model infinite context

But can it actually capture long-range dependencies in practice?

No... due to “vanishing gradients”
梯度消失：因为在很复杂的model，靠近input的layer，在back propegation的时候，几乎接收不到更新的值，所以叫梯度消失

梯度爆炸：当前面靠近output的还可以。但是到靠近input的时候得到一个非常大的一个值，叫做梯度爆炸

Gradients in later steps diminish quickly during backpropagation

Earlier inputs do not get much update

Long Short-term Memory (LSTM)

LSTM is introduced to solve vanishing gradients

Core idea: have “memory cells” that preserve gradients across time

Access to the memory cells is controlled by “gates”

For each input, a gate decides:

‣ how much the new input should be written to the memory cell

‣ and how much content of the current memory cell should be forgotten


Gating Vector
A gate g is a vector
‣ each element has values between 0 to 1

g is multiplied component-wise with vector v, to determine how much information to keep for v

Use sigmoid function to produce g: (使得这个值都是0-1这个区间)
‣ values between 0 to 1

What Are The Disadvantages of LSTM?
Still unable to capture very long range dependencies
Much slower than simple RNNs

Applications

Text Classification
RNNs can be used in a variety of NLP tasks
Particularly suited for tasks where order of words matter, e.g. sentiment classification

Sequence Labelling
Also good for sequence labelling problems, e.g. POS tagging

Variants
 

Multi-layer LSTM
 

Bidirectional LSTM
 

Final Words
Pros
‣ Has the ability to capture long range contexts
‣ Just like feedforward networks: flexible

Cons
‣ Slower than FF networks due to sequential processing
‣ In practice doesn’t capture long range dependency very well (evident when generating very long text)
‣ In practice also doesn’t stack well (multi-layer LSTM)
‣ Less popular nowadays due to the emergence of more advanced architectures (Transformer; lecture 11!)

Lecture 9 (Lexical Semantics(语义))

Definitions
A word sense describes one aspect of the meaning of a word

If a word has multiple senses, it is polysemous

Gloss: textual definition of a sense, given by a dictionary


Meaning Through Relations
Another way to define meaning: by looking at how it relates to other words


Synonymy: near identical meaning

‣ vomit vs. throw up
‣ big vs. large

Antonymy: opposite meaning

‣ long vs. short
‣ big vs. little


Hypernymy: is-a relation

‣ cat is an animal
‣ mango is a fruit


Meronymy: part-whole relation

‣ leg is part of a chair
‣ wheel is part of a car

WordNet

Synsets
Nodes of WordNet are not words or lemmas, but senses

There are represented by sets of synonyms, or synsets

Word Similarity
Synonymy: film vs. movie

Unlike synonymy (which is a binary relation), word similarity is a spectrum(范围)

We can use lexical database (e.g. WordNet) or thesaurus to estimate word similarity

Word Similarity with Paths
Given WordNet, find similarity based on path length

pathlen(c1,c2) = 1+ edge length in the shortest path between sense c1 and c2

Remember that a node in the Wordnet graph is a synset (sense), not a word!

similarity between two senses (synsets)
 
similarity between two words
 

Examples
Each node is a synset! For simplicity we use just the representative word
 
Beyond Path Length
• simpath(nickel,money) = 0.17

• simpath(nickel,Richter scale) = 0.13

• Problem: edges vary widely in actual semantic distance
‣ Much bigger jumps near top of hierarchy

• Solution 1: include depth information (Wu & Palmer)
‣ Use path to find lowest common subsumer (LCS)
‣ Compare using depths

 

Examples
 

Answer:
查看他们的深度来计算
2*1/(6+3) = 2/9 =0.22

Concept Probability Of A Node
 

Example
Abstract nodes higher in the hierarchy has a higher P(c)
 

Similarity with Information Content
 


Word Sense Disambiguation
• Task: selects the correct sense for words in a sentence

• Baseline:
‣ Assume the most popular sense

• Good WSD potentially useful for many tasks
‣ Knowing which sense of mouse is used in a sentence is important!
‣ Less popular nowadays; because sense information is implicitly captured by contextual representations (lecture 11)

Supervised WSD
• Apply standard machine classifiers

• Feature vectors typically words and syntax around target
‣ But context is ambiguous too!
‣ How big should context window be? (in practice small)

• Requires sense-tagged corpora
‣ E.g. SENSEVAL, SEMCOR (available in NLTK)
‣ Very time consuming to create!


Lecture 10 (Distributional Semantics(语义))
Lexical Databases - Problems
• Manually constructed
‣ Expensive
‣ Human annotation can be biased and noisy

• Language is dynamic
‣ New words: slang, terminology, etc.
‣ New senses

• The Internet provides us with massive amounts of text. Can we use that to obtain word meanings?

Word Vectors
• Each row can be thought of a word vector

• It describes the distributional properties of a word
‣ i.e. encodes information about its context words

• Capture all sorts of semantic(语义) relationships (synonymy, analogy, etc)

Word Embeddings?
• We’ve seen word vectors before: word embeddings!

• Here we will learn other ways to produce word vectors
‣ Count-based methods

‣ More efficient neural methods designed just for learning word vectors

 

Count-Based Methods
Learning Count-Based Word Vectors
• Generally two flavours
‣ Use document as context
‣ Use neighbouring words as context

Tf-idf
Standard weighting scheme for information retrieval

Discounts common words!
 


Dimensionality Reduction
• Term-document matrices are very sparse

• Dimensionality reduction: create shorter, denser vectors

• More practical (less features)

• Remove noise (less overfitting)

Singular Value Decomposition (SVD)

 
Truncating – Latent(潜;隐) Semantic(语义) Analysis

• Truncating U, Σ, and VT to k dimensions produces best possible k rank approximation of original matrix

• Uk is a new low dimensional representation of words

• Typical values for k are 100-5000

Words as Context
• Lists how often words appear with other words
‣ In some predefined context (e.g. a window of N words)

• The obvious problem with raw frequency: dominated by common words
‣ But we can’t use tf-idf!


Pointwise Mutual Information
• For two events x and y, PMI computes the
discrepancy between:

‣ Their joint distribution = P(x, y)

‣ Their individual distributions (assuming
independence) = P(x)P(y)
 

Calculating PMI Example

 

PMI Matrix
• PMI does a better job of capturing semantics(语义)
‣ E.g. heaven and hell

• But very biassed towards rare word pairs

• And doesn’t handle zeros well

PMI Tricks

• Zero all negative values (Positive PMI)
‣ Avoid –inf and unreliable negative values

• Counter bias towards rare events

Normalised PMI (PMI(x, y) / −log P(x, y) )

SVD (A = UΣVT)

 

Regardless of whether we use document or word as context, SVD can be applied to create dense vectors

Neural Methods
Word Embeddings
• We’ve seen word embeddings used in neural networks (lecture 7 and 8)

• But these models are designed for other tasks:
‣ Classification
‣ Language modelling

• Word embeddings are just a by-product

Neural Models for Embeddings

• Can we design neural networks whose goal is to purely learn word embeddings?

• Desiderata:
‣ Unsupervised
‣ Efficient

Word2Vec
• Core idea
‣ You shall know a word by the company it keeps
‣ Predict a word using context words

• Framed as learning a classifier
‣ Skip-gram: predict surrounding words of target word
 
‣ Continuous Bag of Words Model (CBOW): predict target word using surrounding words

• Use surrounding words within L positions, L=2 above

Word2Vec limitation
Not good at polysemous(多義的)
For instance, Play and game

Out of word vocabulary(单词表有限，所有都变成<UNK>，丢失信息)

Skip-gram Model
 
 

Embedding parameterisation
 

Training the skip-gram model
• Train to maximise likelihood of raw text

• Too slow in practice, due to normalisation over |V|
 
• Reduce problem to binary classification
‣ (life, rests) → real context word
‣ (aardvark, rests) → non-context word
‣ How to draw non-context word or negative samples?
‣ Randomly from V

Negative Sampling
maximise similarity between target word and real context words

Minimise similarity between target words and non-context words

 

Skip-gram Loss
 
Desiderata
• Unsupervised
‣ Unlabelled corpus

• Efficient
‣ Negative sampling (avoid softmax over full vocabulary)
‣ Scales to very very large corpus

Problems with word vectors/embeddings (count and neural methods)?
Difficult to quantify the quality of word vectors
Don’t capture polysemous(多義的) words

Evaluation

Word Similarity
• Measure similarity of two words using cosine similarity

• Compare predicted similarity with human intuition

• Datasets
‣ WordSim-353 are pairs of nouns with judged relatedness
‣ SimLex-999 also covers verbs and adjectives

Word Analogy(比喻)
• Man is to King as Woman is to ???

• v(Man) - v(King) = v(Woman) - v(???)

• v(???) = v(Woman) - v(Man) + v(King)

• Find word whose embedding is closest to v(Woman) - v(Man) + v(King)

Embedding Space
• Word2Vec embeddings show interesting geometry

• Explains why they are good in word analogy task

 

Downstream(下游) Tasks
• Best evaluation is in other downstream tasks
‣ Use bag-of-word embeddings as a feature representation in a classifier
‣ First layer of most deep learning models is to embed input text
‣ Initialise them with pre-trained word vectors!
 


General Findings
• neural count

• Contextual word representation is shown to work even better

• Dynamic word vectors that change depending on context!

• ELMO & BERT (next lecture!)

Lecture 11 (Contextual Representation)
Word Vectors/Embeddings
• Each word type has one representation
‣ Word2Vec

• Always the same representation regardless of the context of the word

• Does not capture multiple senses of words

• Contextual representation = representation of words based on context

• Pretrained contextual representations work really well for downstream applications!

RNN Language Model
 


Bidirectional RNN
 

Embeddings from Language Models (ELMo)
Trains a bidirectional, multi-layer LSTM language model over 1B word corpus

Combine hidden states from multiple layers of LSTM for downstream tasks
‣ Prior studies use only top layer information

Improves task performance significantly!

Number of LSTM layers = 2

LSTM hidden dimension = 4096

Character convolutional networks (CNN) to create word embeddings
‣ No unknown words
 

Extracting Contextual Representation
 

 

Downstream Task: POS Tagging
 

How Good is ELMO?
This page contains multiple papers baseline result, I don’t think it is useful

Other Findings
• Lower layer representation = captures syntax(句法)
‣ good for POS tagging, NER

• Higher layer representation = captures semantics(语义)
‣ good for QA, textual entailment, sentiment analysis

What are the disadvantages of contextual embeddings?
Difficult to do intrinsic evaluation (e.g. word similarity, analogy)
Interpretability
Computationally expensive to train large-scale contextual embeddings

Bidirectional Encoder Representations from Transformers (BERT)
Disadvantages of RNNs
Sequential processing: difficult to scale to very large corpus or models
RNN language models run left to right (captures only one side of context)
Bidirectional RNNs help, but they only capture surface bidirectional representations

Extracting Contextual Representation
 

BERT: Bidirectional Encoder Representations from Transformers

• Uses self-attention networks (aka Transformers) to capture dependencies between words
‣ No sequential processing

• Masked language model objective to capture deep bidirectional representations

• Loses the ability to generate language

• Not an issue if the goal is to learn contextual representations

 

Objective 1: Masked Language Model
• ‘Mask’ out k% of tokens at random

• Objective: predict the masked words

Objective 2: Next Sentence Prediction
• Learn relationships between sentences

• Predicts whether sentence B follows sentence A

• Useful pre-training objective for downstream applications that analyse sentence pairs (e.g. textual entailment)

Training/Model Details

• WordPiece (subword) Tokenisation

• Multiple layers of transformers to learn contextual representations

• BERT is pretrained on Wikipedia+BookCorpus

• Training takes multiple GPUs over several days

How To Use BERT?
• Given a pretrained BERT, continue training (i.e. fine-tune) it on downstream tasks

• But how to adapt it to downstream task?

• Add a classification layer on top of the contextual representations


BERT vs. ELMo

• ELMo provides only the contextual representations

• Downstream applications has their own network architecture

• ELMo parameters are fixed when applied to downstream applications
‣ Only the weights to combine states from different LSTM layers are learned

• BERT adds a classification layer for downstream tasks
‣ No task-specific model needed

• BERT updates all parameters during fine-tuning


Question: 3 differences and 3 similarities between ELMo and Bert
Similarity:
1.Contextual representation
2.Use left and right context
3. Trained on very large dataset

Difference:
1. 
Bert: transform
ELMo:Bi-LSTM
2.
ELMo: Parameters: fixed
Bert: update all parameters (fine-tune)
3.
BERT: use left and right context at the same time
ELMo: left/right separate


Transformers
What are transformers, and how do they work?
 

Attention is All You Need

Use attention instead of using RNNs (or CNNs) to capture dependencies between words

Self-Attention via Query, Key, Value
• Input:
‣ query q (e.g. made )
‣ key k and value v (e.g. her)

• Query, key and value are all vectors
‣ linear projections from embeddings

• Comparison between query vector of target word (made) and key vectors of context words to compute weights

• Contextual representation of target word = weighted sum of value vectors of context words and target word

 

Self-Attention
 
Multiple queries, stack them in a matrix

Uses scaled dot-product to prevent values from growing too large

Transformer Block
 


Lecture 12 (Discourse)
Discourse
• Most tasks/models we learned operate at word or sentence level:
‣ POS tagging
‣ Language models
‣ Lexical/distributional semantics

• But NLP often deals with documents

• Discourse: understanding how sentences relate to each other in a document

Discourse Segmentation
• A document can be viewed as a sequence of segments

• A segment: a span of cohesive text

• Cohesion: organised around a topic or function

Unsupervised Approaches
• TextTiling algorithm: looking for points of low lexical cohesion between sentences

• For each sentence gap:

‣ Create two BOW vectors consisting of words from k sentences on either side of gap
‣ Use cosine to get a similarity score (sim) for two vectors
‣ For gap i, calculate a depth score, insert boundaries when depth is greater than some threshold t
 

Text Tiling Example (k=1, t=0.9)
 

Supervised Approaches
• Get labelled data from easy sources
‣ Scientific publications
‣ Wikipedia articles

Supervised Discourse Segmenter
• Apply a binary classifier to identify boundaries

• Or use sequential classifiers

• Potentially include classification of section types (introduction, conclusion, etc.)

• Integrate a wider range of features, including
‣ distributional semantics
‣ discourse markers (therefore, and, etc)


Discourse Parsing
Discourse Analysis
• Identify discourse units, and the relations that hold between them

• Rhetorical Structure Theory (RST) is a framework to do hierarchical analysis of discourse structure in documents
 

Discourse Units
• Typically clauses of a sentence

• DUs do not cross sentence boundary

• [It does have beautiful scenery,] [some of the best since Lord of the Rings.]

• 2 merged DUs = another composite DU


Discourse Relations
• Relations between discourse units:

‣ conjuction, justify, concession, elaboration, etc
‣ [It does have beautiful scenery,]
↑(elaboration)
[some of the best since Lord of the Rings.]

Nucleus vs. Satellite
• Within a discourse relation, one argument is the nucleus (the primary argument)

• The supporting argument is the satellite
‣ [It does have beautiful scenery,]nucleus
↑(elaboration)
[some of the best since Lord of the Rings.]satellite

• Some relations are equal (e.g. conjunction), and so both arguments are nuclei
‣ [He was a likable chap,]nucleus
↑(conjunction)
[and I hated to see him die.]nucleus

RST Tree
• An RST relation combines two or more DUs into composite DUs

• Process of combining DUs is repeated, creating an RST tree

 

RST Parsing(解析)
• Task: given a document, recover the RST tree
‣ Rule-based parsing
‣ Bottom-up approach
‣ Top-down approach

Parsing(解析) Using Discourse Markers
• Some discourse markers (cue phrases) explicitly indicate relations
‣ although, but, for example, in other words, so, because, in conclusion,...

• Can be used to build a simple rule-based parser

• However
‣ Many relations are not marked by discourse marker
‣ Many discourse markers ambiguous (e.g. and)

Parsing(解析) Using Machine Learning
• RST Discourse Treebank
‣ 300+ documents annotated with RST trees

• Basic idea:
‣ Segment document into DUs
‣ Combine adjacent DUs into composite DUs iteratively to create the full RST tree (bottom-up parsing)

Bottom-Up Parsing(解析)
• Transition-based parsing (lecture 16):
‣ Greedy, uses shift-reduce algorithm

• CYK/chart parsing algorithm (lecture 14)
‣ Global, but some constraints prevent CYK from finding globally optimal tree for discourse parsing

Top-Down Parsing(解析)
1. Segment document into DUs

2. Decide a boundary to split into 2 segments

3. For each segment, repeat step 2

Discourse Parsing(解析) Features
• Bag of words

• Discourse markers

• Starting/ending n-grams

• Location in the text

• Syntax features

• Lexical and distributional similarities

Applications of Discourse Parsing?
• Summarisation

• Sentiment analysis

• Argumentation

• Authorship attribution

• Essay scoring

Anaphora Resolution
Anaphors
• Anaphor: linguistic expressions that refer back to earlier elements in the text

• Anaphors have a antecedent in the discourse, often but not always a noun phrase
‣ Yesterday, Ted was late for work. It all started when his car wouldn’t start.

• Pronouns are the most common anaphor

• But there are various others
‣ Demonstratives (that problem)

Motivation
• Essential for deep semantic analysis
‣ Very useful for QA, e.g., reading comprehension

Antecedent Restrictions
• Pronouns must agree in number with their antecedents
‣ His coworkers were leaving for lunch when Ted arrived. They invited him, but he said no.

• Pronouns must agree in gender with their antecedents
‣ Sue was leaving for lunch when Ted arrived. She invited him, but he said no.

• Pronouns whose antecedents are the subject of the same syntactic clause must be reflexive (...self)
‣ Ted was angry at him. [him ≠ Ted]
‣ Ted was angry at himself. [himself = Ted]

Antecedent Preferences
• The antecedents of pronouns should be recent
‣ He waited for another 20 minutes, but the tram didn’t come. So he walked home and got his bike out of the garage. He started riding it to work.

• The antecedent should be salient, as determined by grammatical position
‣ Subject > object > argument of preposition
‣ Ted usually rode to work with Bill. He was never late.

Centering Theory
• A unified account of relationship between discourse structure and entity reference

• Every utterance(发声) in the discourse(话语) is characterised by a set of entities(实体), known as centers

• Explains preference of certain entities for ambiguous pronouns

For an Utterance Un
 

Centering Algorithm
• When resolving entity for anaphora resolution, choose the entity such that the top foward-looking center matches with the backward-looking center

• Why? Because the text reads more fluent when this condition is satisfied

Supervised Anaphor Resolution
• Build a binary classifier for anaphor/antecedent pairs

• Convert restrictions and preferences into features
‣ Binary features for number/gender compatibility
‣ Position of antecedent in text
‣ Include features about type of antecedent

• With enough data, can approximate the centering algorithm

• But also easy to include features that are potentially helpful
‣ words around anaphor/antecedent


Lecture 13 (Formal Language Theory & Finite State Automata)

这一节lecture讲的内容感觉很乱，不知道重点在哪里。

Regular Language
• The simplest class of languages

• Any regular expression is a regular language

‣ Describes what strings are part of the language (e.g. ‘0(0|1)*1’)

Properties of Regular Languages
• Closure: if we take regular languages L1 and L2 and merge them, is the resulting language regular?

• RLs are closed under the following:
‣ concatenation and union
‣ intersection: strings that are valid in both L1 and L2
‣ negation: strings that are not in L

• Extremely versatile! Can have RLs for different properties of language, and use them together

Finite State Acceptor
• Regular expression defines a regular language

• But it doesn’t give an algorithm to check whether a string belongs to the language

• Finite state acceptors (FSA) describes the computation involved for membership checking

• FSA consists:
‣ alphabet of input symbols, Σ
‣ set of states, Q
‣ start state, q0 ∈ Q
‣ final states, F ⊆ Q
‣ transition function: symbol and state → next state

• Accepts strings if there is path from q0 to a final state with transitions matching each symbol
‣ Djisktra’s shortest-path algorithm, O(V log V + E)

Derivational Morphology
• Use of affixes to change word to another grammatical category

• grace → graceful → gracefully

• grace → disgrace → disgracefully

• allure → alluring → alluringly

• allure → *allureful

• allure → *disallure

FSA for Morphology
• Fairly consistent process

‣ want to accept valid forms (grace → graceful)

‣ reject invalid ones (allure → *allureful)

‣ generalise to other words, e.g., nouns that behave like grace or allure


Weighted FSA
• Some words are more plausible than others

‣ fishful vs. disgracelyful

‣ musicky vs. writey

• Graded measure of acceptability — weighted FSA
changes the following:

‣ start state weight function, λ: Q → R

‣ final state weight function, ρ: Q → R

‣ transition function, δ: (Q, Σ, Q) → R

Finite State Transducer
• Often don’t want to just accept or score strings
‣ want to translate them into another string

• FST add string output capability to FSA
‣ includes an output alphabet
‣ transitions now take input symbol and emit output symbol (Q, Σ, Σ, Q)

• Can be weighted = WFST
‣ Graded scores for transition

• E.g., edit distance as WFST
‣ distance to transform one string to another

Lecture 14 (Context-Free Grammar)
Basics of Context-Free Grammars
• Symbols
‣ Terminal: word such as book
‣ Non-terminal: syntactic label such as NP or VP

• Productions (rules)
‣ W → X Y Z
‣ Exactly one non-terminal on left-hand side (LHS)
‣ An ordered list of symbols on right-hand side (RHS); can be terminals or non-terminals

• Start symbol: S

Context-Free vs. Regular
• Context-free languages more general than regular languages
‣ Allows recursive nesting
 
CFG Parsing
• Given production rules
‣ S → a S b
‣ S → a b

• And a string
‣ aaabbb

• Produce a valid parse tree
 
What This Means?
• If English can be represented with CFG:
‣ first develop the production rules
‣ can then build a “parser” to automatically judge whether a sentence is grammatical!

• But is natural language context-free?

• Not quite: cross-serial dependencies (ambncmdn)

But…

• CFG strike a good balance:
‣ CFG covers most syntactic patterns
‣ CFG parsing is computational efficient

• We use CFG to describe a core fragment of English syntax

Constituents
Syntactic Constituents
• Sentences are broken into constituents
‣ word sequence that function as a coherent unit for linguistic analysis
‣ helps build CFG production rules

• Constituents have certain key properties:
‣ movement
‣ substitution
‣ coordination

Movement
• Constituents can be moved around sentences
‣ Abigail gave [her brother] [a fish]
‣ Abigail gave [a fish] to [her brother]

• Contrast: [gave her], [brother a]

Substitution
• Constituents can be substituted by other phrases of the same type
‣ Max thanked [his older sister]
‣ Max thanked [her]

• Contrast: [Max thanked], [thanked his]

Coordination
• Constituents can be conjoined with coordinators like and and or
‣ [Abigail] and [her young brother] brought a fish
‣ Abigail [bought a fish] and [gave it to Max]
‣ Abigail [bought] and [greedily ate] a fish

Constituents and Phrases
• Once we identify constituents, we use phrases to describe them

• Phrases are determined by their head word:
‣ noun phrase: her younger brother
‣ verb phrase: greedily ate it

• We can use CFG to formalise these intuitions

A Simple CFG for English
Terminal symbols: rat, the, ate, cheese

Non-terminal symbols: S, NP, VP, DT, VBD, NN

Productions:
S → NP VP
NP → DT NN
VP → VBD NP
DT → the
NN → rat
NN → cheese
VBD → ate

CFG Trees
• Generation corresponds to a syntactic tree
• Non-terminals are internal nodes

• Terminals are leaves

• CFG parsing is the reverse process (sentence → tree)
 
CYK Algorithm
• Bottom-up parsing(解析)

• Tests whether a string is valid given a CFG, without enumerating all possible parses

• Core idea: form small constituents first, and merge them into larger constituents

• Requirement: CFGs must be in Chomsky Normal Forms

Convert to Chomsky Normal Form
• Change grammar so all rules of form:
‣ A → B C
‣ A → a

• Convert rules of form A → B c into:
‣ A → B X
‣ X → c

• Convert rules A → B C D into:
‣ A → B Y
‣ Y → C D
‣ E.g. VP → VP NP NP for ditransitive cases, “sold [her] [the book]”

• X, Y are new symbols we have introduced

• CNF disallows unary rules, A → B.

• Imagine NP → S; and S → NP ... leads to infinitely many trees with same yield.

• Replace RHS non-terminal with its productions

• A → B, B → cat, B → dog

• A → cat, A → dog

The CYK Parsing Algorithm
• Convert grammar to Chomsky Normal Form (CNF)

• Fill in a parse table (left to right, bottom to top)

• Use table to derive parse

• S in top right corner of table = success!

• Convert result back to original grammar

Example
There are two ways to represent this CYK
 

Picture 2:
 
Picture 3:
 

CYK Algorithm
 


Representing English with CFGs

From Toy Grammars to Real Grammars
• Toy grammars with handful of productions good for demonstration or extremely limited domains

• For real texts, we need real grammars

• Many thousands of production rules

Key Constituents in Penn Treebank

• Sentence (S)

• Noun phrase (NP)

• Verb phrase (VP)

• Prepositional phrase (PP)

• Adjective phrase (AdjP)

• Adverbial phrase (AdvP)

• Subordinate clause (SBAR)



Lecture 15 (Context-Free Grammar)
Ambiguity In Parsing
• Context-free grammars assign hierarchical structure to language
‣ Formulated as generating all strings in the language
‣ Predicting the structure(s) for a given string

• Raises problem of ambiguity — which is
better?

• Probabilistic CFG!

Basics of PCFGs
• Same symbol set:
‣ Terminals: words such as book
‣ Non-terminal: syntactic labels such as NP or NN

• Same productions (rules)
‣ LHS non-terminal → ordered list of RHS symbols

• In addition, store a probability with each production
‣ NP → DT NN [p = 0.45]
‣ NN → cat [p = 0.02]
‣ NN → leprechaun [p = 0.00001]

• Probability values denote conditional
‣ P(LHS → RHS)
‣ P(RHS | LHS)

• Consequently they:
‣ must be positive values, between 0 and 1
‣ must sum to one for given LHS

• E.g.,
‣ NN → aadvark [p = 0.0003]
‣ NN → cat [p = 0.02]
‣ NN → leprechaun [p = 0.0001]
‣ ∑ P(NN → x) = 1

Stochastic Generation with PCFGs
Almost the same as for CFG, with one twist:

1. Start with S, the sentence symbol

2. Choose a rule with S as the LHS
‣ Randomly select a RHS according to P(RHS | LHS) e.g., S → VP
‣ Apply this rule, e.g., substitute VP for S

3. Repeat step 2 for each non-terminal in the string (here, VP)

4. Stop when no non-terminals remain Gives us a tree, as before, with a sentence as the yield

How Likely Is a Tree?
• Given a tree, we can compute its probability
‣ Decomposes into probability of each production

• P(tree) =
P(S → VP) ×
P(VP → Verb NP) ×
P(Verb → Book) ×
P(NP → Det Nominal) ×
P(Det → the) ×
P(Nominal → Nominal Noun) ×
P(Nominal → Noun) ×
P(Noun → dinner) ×
P(Noun → flight)

Resolving Parse Ambiguity
• Can select between different trees based on P(T)

• P(Tleft) = 2.2 × 10-6
P(Tright) = 6.1 × 10-7

PCFG Parsing
Parsing PCFGs
• Before we looked at

‣ CYK
‣ for unweighted grammars (CFGs)
‣ finds all possible trees

• But there are often 1000s, many completely nonsensical

• Can we solve for the most probable tree?

CYK for PCFGs
• CYK finds all trees for a sentence; we want best tree

• Prob. CYK follows similar process to standard CYK

• Convert grammar to Chomsky Normal Form (CNF)

‣ VP → Verb NP NP [0.10]

‣ VP → Verb NP+NP [0.10]
NP+NP → NP NP [1.0]

‣ where NP+NP is a new symbol.

Example

 

Prob CYK: Retrieving the Parses
• S in the top-right corner of parse table indicates success

• Retain back-pointer to best analysis

• To get parse(s), follow pointers back for each match

• Convert back from CNF by removing new non-terminals

Prob. CYK
 

Limitations of CFG
CFG Problem 1: Poor Independence Assumptions
• Rewrite decisions made independently, whereas inter-dependence is often needed to capture global structure.

• NP → DT NN [0.28]

• NP → PRP [0.25]

• Probability of a rule independent of rest of tree

• No way to represent this contextual differences in PCFG probabilities

Poor Independence Assumptions
 

• NP → PRP should go up to 0.91 as a subject

• NP → DT NN should be 0.66 as an object

• Solution: add a condition to denote whether NP is a subject or object

Solution: Parent Conditioning
• Make non-terminals more explicit by incorporating parent symbol into each symbol
 

• NP^S represents subject position (left)
• NP^VP denotes object position (right)

CFG Problem 2: Lack of Lexical Conditioning
• Lack of sensitivity to words in tree

• Prepositional phrase (PP) attachment ambiguity
‣ Worker dumped sacks into a bin
 
PP Attachment Ambiguity
 

Coordination Ambiguity
• dogs in houses and cats
 

• dogs is semantically a better conjunct for cats than houses (dogs can’t fit into cats!)

Solution: Head Lexicalisation
• Record head word with parent symbols

‣ the most salient child of a constituent, usually the noun in a NP, verb in a VP etc

‣ VP → VBD NP PP => VP(dumped) → VBD(dumped) NP(sacks) PP(into)

Head Lexicalisation
• Incorporate head words into productions, to capture the most important links between words
‣ Captures correlations between head words of phrases
‣ PP(into): VP(dumped) vs. NP(sacks)

• Grammar symbol inventory expands massively!
‣ Many of the productions too specific, rarely seen
‣ Learning more involved to avoid sparsity problems (e.g., zero probabilities)

Lecture 16 (Dependency Grammar)
Dependency Grammars
• Dependency grammar offers a simpler approach
‣ describe relations between pairs of words
‣ namely, between heads and dependents
‣ e.g. (prefer, dobj, flight)
Why?
• Deal better with languages that are morphologically(形态上) rich and have a relatively free word order

‣ CFG need a separate rule for each possible place a phrase can occur in

• Head-dependent relations similar to semantic relations between words
‣ More useful for applications: coreference resolution, information extraction, etc

Basics of Dependency Grammar
Dependency Relations
• Captures the grammatical relation between:
‣ Head = central word
‣ Dependent = supporting word

• Grammatical relation = subject, direct object, etc

• Many dependency theories and taxonomies proposed for different languages

• Universal Dependency: a framework to create a set of dependency relations that are computationally useful and cross-lingual

Question Answering
• Dependency tree more directly represents the core of the sentence: who did what to whom?
‣ captured by the links incident on verb nodes

Information Extraction
• “Brasilia, the Brazilian capital, was founded in 1960.”
→ capital(Brazil, Brasilia)
→ founded(Brasilia, 1960)

• Dependency tree captures relations succinctly
 


What about CFGs
• Constituency trees can also provide similar information

• But it requires some distilling(蒸馏) using head-finding rules
 

Dependency vs Constituency
• Dependency tree
‣ each node is a word token
‣ one node is chosen as the root
‣ directed edges link heads and their dependents

• Constituency tree
‣ forms a hierarchical tree
‣ word tokens are the leaves
‣ internal nodes are ‘constituent phrases’ e.g. NP

• Both use part-of-speech

Properties of a Dependency Tree
• Each word has a single head (parent)

• There is a single root node

• There is a unique path to each word from the root

• All arcs should be projective

Projectivity
• An arc is projective if there is a path from head to every word that lies between the head and the dependent

• Dependency tree is projective if all arcs are projective

• In other words, a dependency tree is projective if it can be drawn with no crossing edges

• Most sentences are projective, but exceptions exist

• Common in languages with flexible word order

Treebank Conversion
• A few dependency treebanks (Czech, Arabic, Danish...)

• Many constituency treebanks

• Some can be converted into dependencies

• Dependency trees generated from constituency trees are always projective

• Main idea: identify head-dependent relations in constituency structure and the corresponding dependency relations
‣ Use various heuristics, e.g., head-finding rules
‣ Often with manual correction

Examples from treebanks
• Danish DDT includes additional ‘subject’ link for verbs

• METU-Sabancı Turkish treebank
‣ edges between morphological units, not just words
 

Transition-based Parsing
Dependency Parsing
• Find the best structure for a given input sentence

• Two main approaches:

‣ Transition-based: bottom-up greedy method

‣ Graph-based: encodes problem using nodes/ edges and use graph theory methods to find optimal solutions

Caveat
• Transition-based parsers can only handle projective dependency trees!

• Less applicable for languages where cross-dependencies are common

Transition-Based Parsing: Intuition
• Processes word from left to right

• Maintain two data structures
‣ Buffer: input words yet to be processed
‣ Stack: store words that are being processed

• At each step, perform one of the 3 actions:
‣ Shift: move a word from buffer to stack
‣ Left-Arc: assign current word as head of the previous word in stack
 
‣ Right-Arc: assign previous word as head of current word in stack
 

Dependency Labels
• For simplicity, we omit labels on the dependency relations

• In practice, we parameterise the left-arc and right-arc actions with dependency labels:
‣ E.g. left-arc-nsubj or right-arc-dobj

• Expands the list of actions to > 3 types

The Right Action?
• We assume an oracle(神谕) that tells us the right action at every step

• Given a dependency tree, the role of oracle is to generate a sequence of ground truth actions

Parsing Model
• We then train a supervised model to mimic the actions of the oracle

‣ To learn at every step the correct action to take (as given by the oracle)

‣ At test time, the trained model can be used to parse a sentence to create the dependency tree

Parsing As Classification
• Input:
‣ Stack (top-2 elements: s1 and s2)
‣ Buffer (first element: b1)

• Output
‣ 3 classes: shift, left-arc, or, right-arc

• Features
‣ word (w), part-of-speech (t)

 

Classifiers
• Traditionally SVM works best

• Nowadays, deep learning models are state-of-the-art

• Weakness: local classifier based on greedy search

• Solutions:
‣ Beam search: keep track of top-N best actions
‣ Dynamic oracle: during training, use predicted actions occasionally
‣ Graph-based parser

Graph-based Parsing
• Given an input sentence, construct a fully-connected, weighted, directed graph

• Vertices: all words

• Edges: head-dependent arcs

• Weight: score based on training data (relation that is frequently observed receive a higher score)

• Objective: find the maximum spanning tree (Kruskal’s algorithm)

Advantage
• Can produce non-projective trees
‣ Not a big deal for English
‣ But important for many other languages

• Score entire trees
‣ Avoid making greedy local decisions like transition-based parsers
‣ Captures long dependencies better

Example
 

• Caveat: tree may contain cycles

• Solution: need to do cleanup to remove cycles (Chu-Liu-Edmonds algorithm)

Lecture 17 (Machine Translation)
Introduction
Machine translation (MT) is the task of translating text from one source language to another target language

Why?
• Removes language barrier

• Makes information in any languages accessible to anyone

• But translation is a classic “AI-hard” challenge
‣ Difficult to preserve the meaning and the fluency of the text after translation

MT is Difficult
• Not just simple word for word translation

• Structural changes, e.g., syntax and semantics

• Multiple word translations, idioms

• Inflections for gender, case etc

• Missing information (e.g., determiners)

Statistical MT
Early Machine Translation
• Started in early 1950s

• Motivated by the Cold War to translate Russian to English

• Rule-based system

‣ Use bilingual dictionary to map Russian words to English words

• Goal: translate 1-2 million words an hour within 5 years
Statistical MT
• Given French sentence f, aim is to find the best
English sentence e

‣argmaxe P(e |f)

• Use Baye’s rule to decompose into two components

‣argmaxe P(f | e)P(e)

Language vs Translation Model
argmaxe P(f | e)P(e)

• P(e): language model
‣ learns how to write fluent English text

• P(f|e): translation model
‣ learns how to translate words and phrases from English to French

How to Learn Language Model (LM) and Translation Model (TM)?
• Language model:
‣ Text statistics in large monolingual corpora

• Translation model:
‣ Word co-occurrences in parallel corpora
‣ i.e. English-French sentence pairs

Parallel Corpora
• One text in multiple languages

• Produced by human translation
‣ Bible, news articles, legal transcripts, literature, subtitles

‣ Open parallel corpus: http://opus.nlpl.eu/
 

Models of Translation
• How to learn P(f|e) from parallel text?

• We only have sentence pairs; words are not aligned in the parallel text

• I.e. we don’t have word to word translation
 

Alignment
• Idea: introduce word alignment as a latent variable into the model

‣P(f, a| e)

• Use algorithms such as expectation maximisation (EM) to learn (e.g. GIZA++)

 

Complexity of Alignment

Some words are dropped and have no alignment
 

One-to-many alignment
 
Many-to-one alignment
 
Many-to-many alignment
 
Statistical MT: Summary
• A very popular field of research in NLP prior to 2010s

• Lots of feature engineering

• State-of-the-art systems are very complex
‣ Difficult to maintain
‣ Significant effort needed for new language pairs

Neural Machine Translation
Introduction
• Neural machine translation is a new approach to do machine translation

• Use a single neural model to directly translate from source to target

• Requires parallel text

• Architecture: encoder-decoder model
‣ 1st RNN to encode the source sentence
‣ 2nd RNN to decode the target sentence

 
Neural MT
• The decoder RNN can be interpreted as a conditional language model

‣ Language model: predicts the next word given previous words in target sentence y

‣ Conditional: prediction is also conditioned on the source sentence x

• P(y | x) = P(y1 | x)P(y2 | y1, x) . . . P(yt| y1, . . . , yt−1, x)

Training Neural MT
• Requires parallel corpus just like statistical MT

• Trains with next word prediction, just like a language model!

Language Model Training Loss
 

Neural MT Training Loss
 

Training
 

• During training, we have the target sentence

• We can therefore feed the right word from target sentence, one step at a time

Decoding at Test Time
 
• But at test time, we don’t have the target sentence (that’s what we’re trying to predict!)

• argmax: take the word with the highest probability at every step

Exposure Bias
 
• Describes the discrepancy(差异) between training and testing

• Training: always have the ground truth tokens at each step

• Test: uses its own prediction at each step

• Outcome: model is unable to recover from its own error

Greedy Decoding
• argmax decoding is also called greedy decoding

• Issue: does not guarantee optimal probability P(y | x)

Exhaustive Search Decoding
• To find optimal, we need to consider every word at every step to compute the probability of all possible sequences

• O(Vn); V = vocab size; n = sentence length

• Far too expensive to be feasible

Beam Search Decoding
• Instead of considering all possible words at every step, consider k best words

• That is, we keep track of the top-k words that produce the best partial translations (hypotheses) thus far

• k = beam width (typically 5 to 10)

• k = 1 = greedy decoding

• k = V = exhaustive search decoding

Example
 

When to Stop?
• When decoding, we stop when we generate <end> token

• But multiple hypotheses may terminate their sentence at different time steps

• We store hypotheses that have terminated, and continue explore those that haven’t

• Typically we also set a maximum sentence length that can be generated (e.g. 50 words)

What are some issues of NMT?

• Information of the whole source sentence is represented by a single vector
• NMT can generate new details not in source sentence
• Black-box model; difficult to explain when it doesn’t work

Neural MT: Summary
• Single end-to-end model
‣ Statistical MT systems have multiple sub-components

• Less feature engineering

• Can produce new details that are not in the source sentence (hallucination)

Attention Mechanism
 

• With a long source sentence, the encoded vector is unlikely to capture all the information in the sentence

• This creates an information bottleneck

Attention
• For the decoder, at every time step allow it to ‘attend’ to words in the source sentence
 

Encoder-Decoder with Attention
 

Variants
 

Attention: Summary
• Solves the information bottleneck issue by allowing decoder to have access to the source sentence words directly

• Provides some form of interpretability
‣ Attention weights can be seen as word alignments

• Most state-of-the-art Neural Machine Translation (NMT) systems use attention
- Google Translate

Evaluation
Machine Translation Evaluation
• BLEU: compute n-gram overlap between “reference” translation and generated translation

• Typically computed for 1 to 4-gram
 

Lecture 18 (Information Extraction)
• Given this:
‣ “Brasilia, the Brazilian capital, was founded in 1960.”

• Obtain this:
‣ capital(Brazil, Brasilia)
‣ founded(Brasilia, 1960)

• Main goal: turn text into structured data

Applications
• Stock analysis
‣ Gather information from news and social media
‣ Summarise texts into a structured format
‣ Decide whether to buy/sell at current stock price

• Medical research
‣ Obtain information from articles about diseases and treatments
‣ Decide which treatment to apply for new patient

How?
• Given this:
‣ “Brasilia, the Brazilian capital, was founded in 1960.”

• Obtain this:
‣ capital(Brazil, Brasilia)
‣ founded(Brasilia, 1960)

• Two steps:
‣ Named Entity Recognition (NER): find out entities such as “Brasilia” and “1960”
‣ Relation Extraction: use context to find the relation between “Brasilia” and “1960” (“founded”)

Machine learning in IE
• Named Entity Recognition (NER): sequence models such as RNNs, HMMs or CRFs.

• Relation Extraction: mostly classifiers, either binary or multi-class.

• This lecture: how to frame these two tasks in order to apply sequence labellers and classifiers.

Named Entity Recognition

Typical Entity Tags
• PER: people, characters

• ORG: companies, sports teams

• LOC: regions, mountains, seas

• GPE: countries, states, provinces (in some tagset this is labelled as LOC)

• FAC: bridges, buildings, airports

• VEH: planes, trains, cars

• Tag-set is application-dependent: some domains deal with specific entities e.g. proteins and genes

NER as Sequence Labelling
• NE tags can be ambiguous:
‣ “Washington” can be a person, location or political entity

• Similar problem when doing POS tagging
‣ Incorporate context

• Can we use a sequence tagger for this (e.g. HMM)?
‣ No, as entities can span multiple tokens
‣ Solution: modify the tag set

IO tagging
• [ORG American Airlines], a unit of [ORG AMR Corp.], immediately matched the move, spokesman [PER Tim Wagner] said.
• ‘I-ORG’ represents a token that is inside an entity (ORG in this case).
• All tokens which are not entities get the ‘O’ token (for outside).

• Cannot differentiate between:

‣ a single entity with multiple tokens

‣ multiple entities with single tokens

IOB tagging
B-ORG represents the beginning of an ORG entity.

If the entity has more than one token, subsequent tags are represented as I-ORG.

NER as Sequence Labelling
• Given such tagging scheme, we can train any sequence labelling model

• In theory, HMMs can be used but discriminative models such as CRFs are preferred

Relation Extraction
Methods
• If we have access to a fixed relation database:
‣ Rule-based
‣ Supervised
‣ Semi-supervised
‣ Distant supervision

• If no restrictions on relations:
‣ Unsupervised
‣ Sometimes referred as “OpenIE”

Rule-Based Relation Extraction
• “Agar is a substance prepared from a mixture of red algae such as Gelidium, for laboratory or industrial use.”

• [NP red algae] such as [NP Gelidium]

• NP0 such as NP1 → hyponym(NP1, NP0)

• hyponym(Gelidium, red algae)

• Lexico-syntactic patterns: high precision, low recall, manual effort required

Supervised Relation Extraction
• Assume a corpus with annotated relations

• Two steps. First, find if an entity pair is related or not (binary classification)
‣ For each sentence, gather all possible entity pairs
‣ Annotated pairs are considered positive examples
‣ Non-annotated pairs are taken as negative examples

• Second, for pairs predicted as positive, use a multi-class classifier (e.g. SVM) to obtain the relation

• First:
‣ (American Airlines, AMR Corp.) → positive
‣ (American Airlines, Tim Wagner) → positive
‣ (AMR Corp., Tim Wagner) → negative

• Second:
‣ (American Airlines, AMR Corp.) → subsidiary
‣ (American Airlines, Tim Wagner) → employment

Semi-supervised Relation Extraction
• Annotated corpora is very expensive to create

• Use seed tuples to bootstrap a classifier

1. Given seed tuple: hub(Ryanair, Charleroi)

2. Find sentences containing terms in seed tuples
• Budget airline Ryanair, which uses Charleroi as a hub, scrapped all weekend flights out of the airport.

3. Extract general patterns
• [ORG], which uses [LOC] as a hub

4. Find new tuples with these patterns
• hub(Jetstar, Avalon)

5. Add these new tuples to existing tuples and repeat step 2

What are some issues of such semi-supervised relation extraction method?
Extracted tuples deviate from original relation over time

Difficult to evaluate

Extracted general patterns tend to be very noisy

Semantic Drift
• Pattern: [NP] has a {NP}* hub at [LOC]

• Sydney has a ferry hub at Circular Quay
‣ hub(Sydney, Circular Quay)

• More erroneous patterns extracted from this tuple...

• Should only accept patterns with high confidences

Distant Supervision
• Semi-supervised methods assume the existence of seed tuples to mine new tuples

• Can we mine new tuples directly?

• Distant supervision obtain new tuples from a range of sources:
‣ DBpedia
‣ Freebase
 
• Generate massive training sets, enabling the use of richer features, and no risk of semantic drift

Unsupervised Relation Extraction (“OpenIE”)
• No fixed or closed set of relations

• Relations are sub-sentences; usually has a verb

• “United has a hub in Chicago, which is the headquarters of United Continental Holdings.”
‣ “has a hub in”(United, Chicago)
‣ “is the headquarters of”(Chicago, United Continental Holdings)

• Main problem: mapping relations into canonical forms

Evaluation
• NER: F1-measure at the entity level.

• Relation Extraction with known relation set: F1-measure

• Relation Extraction with unknown relations: much harder to evaluate
‣ Usually need some human evaluation
‣ Massive datasets used in these settings are impractical to evaluate manually (use samples)
‣ Can only obtain (approximate) precision, not recall.

Other IE Tasks
Temporal Expression Extraction
• Anchoring: when is “last week”?
‣ “last week” → 2007−W26

• Normalisation: mapping expressions to canonical forms.
‣ July 2, 2007 → 2007-07-02

• Mostly rule-based approaches

Event Extraction
• Very similar to NER, including annotation and learning methods.

• Event ordering: detect how a set of events happened in a timeline.
‣ Involves both event extraction and temporal expression extraction.

Lecture 19 (Question Answering)
Definition: question answering (“QA”) is the task of automatically determining the answer for a natural language question

Mostly focus on “factoid” questions

Factoid Questions
没有什么内容
Non-factoid Questions
没有什么内容

Why do we focus on factoid questions in NLP?
• They are easier

• They have an objective answer

• Current NLP technologies cannot handle non-factoid answers

• There’s less demand for systems to automatically answer non-factoid questions

2 Key Approaches
• Information retrieval-based QA
‣ Given a query, search relevant documents
‣ Find answers within these relevant documents

• Knowledge-based QA
‣ Builds semantic(语义) representation of the query
‣ Query database of facts to find answers

IR-based QA (Information retrieval)
IR-based Factoid QA: TREC-QA

1. Use question to make query for IR engine

2. Find document, and passage within document

3. Extract short answer string

Question Processing
• Find key parts of question that will help retrieval
‣ Discard non-content words/symbols (wh-word, ?, etc)
‣ Formulate as tf-idf query, using unigrams or bigrams
‣ Identify entities and prioritise match

• May reformulate question using templates
‣ E.g. “Where is Federation Square located?”
‣ Query = “Federation Square located”
‣ Query = “Federation Square is located [in/at]”

• Predict expected answer type (here = LOCATION)

Answer Types
• Knowing the type of answer can help in:
‣ finding the right passage containing the answer
‣ finding the answer string

• Treat as classification
‣ given question, predict answer type
‣ key feature is question headword
‣ What are the animals on the Australian coat of arms?
‣ Generally not a difficult task

Retrieval
• Find top n documents matching query (standard IR)

• Next find passages (paragraphs or sentences) in these documents (also driven by IR)

• Should contain:
‣ many instances of the question keywords
‣ several named entities of the answer type
‣ close proximity of these terms in the passage
‣ high ranking by IR engine

• Re-rank IR outputs to find best passage (e.g., using supervised learning)

Answer Extraction
• Find a concise answer to the question, as a span(跨度) in the passage

How?
• Use a neural network to extract answer

• AKA reading comprehension task

• But deep learning models require lots of data

• Do we have enough data to train comprehension models?

SQuAD
• Use Wikipedia passages

• First set of crowdworkers create questions (given passage)

• Second set of crowdworkers label the answer

• 150K questions (!)

• Second version includes unanswerable questions

Knowledge-based QA
QA over structured KB
• Many large knowledge bases
‣ Freebase, DBpedia, Yago, ...

• Can we support natural language queries?
‣ E.g.
‣ Link “Ada Lovelace” with the correct entity in the
KB to find triple (Ada Lovelace, birth-year, 1815)

But…
• Converting natural language sentence into triple is not trivial

• Entity linking also an important component
‣ Ambiguity: “When was Lovelace born?”

• Can we simplify this two-step process?

Semantic Parsing(语义解析)
• Convert questions into logical forms to query KB directly

‣ Predicate calculus

‣ Programming query (e.g. SQL)

Hybrid QA

Hybrid Methods
• Why not use both text-based and knowledge-based resources for QA?

• IBM’s Watson which won the game show Jeopardy(危险)! uses a wide variety of resources to answer questions
‣ THEATRE: A new play based on this Sir Arthur Conan Doyle canine classic opened on the London stage in 2007.
‣ The Hound Of The Baskervilles

Core Idea of Watson

• Generate lots of candidate answers from text-based and knowledge-based sources

• Use a rich variety of evidence to score them

• Many components in the system, most trained separately

QA Evaluation
• IR(Information retrieval): Mean Reciprocal Rank for systems returning matching passages or answer strings

‣ E.g. system returns 4 passages for a query, first correct passage is the 3rd passage

‣ MRR = 1⁄3

• MCTest: Accuracy

• SQuAD: Exact match of string against gold answer


Lecture 20 (Topic Modelling)
Applications of topic models?
• Personalised advertising

• Search engine

• Discover senses of polysemous words

• Part-of-speech tagging

A Brief History of Topic Models
Latent Semantic Analysis (L10)
 
LSA: Truncate

Issues
• Positive and negative values in the U and VT

• Difficult to interpret

Probabilistic LSA
Based on a probabilistic model
 

Issues
• No more negative values!

• PLSA can learn topics and topic assignment for documents in the train corpus

• But it is unable to infer topic distribution on new documents

• PLSA needs to be re-trained for new documents

Latent Dirichlet Allocation
• Introduces a prior to the document-topic and topic-word distribution

• Fully generative: trained LDA model can infer topics on unseen documents!

• LDA is a Bayesian version of PLSA

Latent Dirichlet Allocation
• Core idea: assume each document contains a mix of topics

• But the topic structure is hidden (latent)

• LDA infers the topic structure given the observed words and documents

• LDA produces soft clusters of documents (based on topic overlap), rather than hard clusters

• Given a trained LDA model, it can infer topics on new documents (not part of train data)

Input
• A collection of documents

• Bag-of-words

• Good preprocessing practice:
‣ Remove stopwords
‣ Remove low and high frequency word types
‣ Lemmatisation

Output
• Topics: distribution over words in each topic

• Topic assignment (任务): distribution over topics in each document

Learning
• How do we learn the latent topics?

• Two main family of algorithms:
‣ Variational methods
‣ Sampling-based methods

Sampling Method (Gibbs)
1. Randomly assign topics to all tokens in documents

2. Collect topic-word and document-topic co-occurrence statistics based on the assignments

When Do We Stop?
• Train until convergence

• Convergence = model probability of training set becomes stable

• How to compute model probability?
 

Hyper-Parameters
T: number of topic

• alpha: prior on the topic-word distribution

• belta: prior on the document-topic distribution

• Analogous to k in add-k smoothing in N-gram LM

• Pseudo counts to initialise co-occurrence matrix:

Evaluation
How To Evaluate Topic Models?
• Unsupervised learning → no labels

• Intrinsic evaluation:
‣ model logprob / perplexity on test documents

 
Issues with Perplexity(困惑)
• More topics = better (lower) perplexity

• Smaller vocabulary = better perplexity
‣ Perplexity not comparable for different corpora, or different tokenisation/preprocessing methods

• Does not correlate with human perception of topic quality

• Extrinsic evaluation the way to go:
‣ Evaluate topic models based on downstream task

Topic Coherence
• A better intrinsic evaluation method

• Measure how coherent the generated topics

• A good topic model is one that generates more coherent topics

Word Intrusion
• Idea: inject one random word to a topic

{farmers, farm, food, rice, agriculture}

↓

{farmers, farm, food, rice, cat, agriculture}

• Ask users to guess which is the intruder word

• Correct guess → topic is coherent

• Try guess the intruder word in:
‣ {choice, count, village, i.e., simply, unionist}

• Manual effort; does not scale

PMI ≈ Coherence?
• High PMI for a pair of words → words are correlated
‣ PMI(farm, rice) ↑
‣ PMI(choice, village) ↓

• If all word pairs in a topic has high PMI → topic is coherent

• If most topics have high PMI → good topic model

• Where to get word co-occurrence statistics for PMI?

‣ Can use same corpus for topic model
‣ A better way is to use an external corpus (e.g. Wikipedia)

PMI
Compute pairwise PMI of top-N words in a topic
 
Given topic: {farmers, farm, food, rice, agriculture}
Coherence = sum PMI for all word pairs:
‣ PMI(farmers, farm) + PMI(farmers, food) + ... + PMI(rice, agriculture)

Variants
• Normalised PMI
 
• Conditional probability
 

Lecture 21 (Summarisation)
Summarisation
• Distill the most important information from a text to produce shortened or abridged version

• Examples
‣ outlines of a document
‣ abstracts of a scientific article
‣ headlines of a news article
‣ snippets of search result

What to Summarise?
• Single-document summarisation
‣ Input: a single document
‣ Output: summary that characterise the content

• Multi-document summarisation
‣ Input: multiple documents

‣ Output: summary that captures the gist of all documents
‣ E.g. summarise a news event from multiple sources or perspectives

How to Summarise?
• Extractive summarisation
‣ Summarise by selecting representative sentences from documents

• Abstractive summarisation
‣ Summarise the content in your own words
‣ Summaries will often be paraphrases of the original content

Goal of Summarisation?
• Generic summarisation
‣ Summary gives important information in the document(s)

• Query-focused summarisation
‣ Summary responds to a user query
‣ "Non-factoid" QA
‣ Answer is much longer than factoid QA

Query-Focused Summarisation

Extractive: Single-Doc
Summarisation System
• Content selection: select what sentences to extract from the document

• Information ordering: decide how to order extracted sentences

• Sentence realisation: cleanup to make sure combined sentences are fluent

• We will focus on content selection

• For single-document summarisation, information ordering not necessary
‣ present extracted sentences in original order

• Sentence realisation also not necessary if they are presented in dot points
Content Selection
• Not much data with ground truth extractive sentences

• Mostly unsupervised methods

• Goal: Find sentences that are important or salient

Method 1: TF-IDF
• Frequent words in a doc → salient

• But some generic words are very frequent but uninformative
‣ function words
‣ stop words

• Weigh each word w in document d by its inverse document frequency:
‣ 

Method 2: Log Likelihood Ratio
Intuition: a word is salient if its probability in the input corpus is very different to a background corpus
 
Saliency of A Sentence?
 

Method 3: Sentence Centrality
• Alternative approach to ranking sentences

• Measure distance between sentences, and choose sentences that are closer to other sentences

• Use tf-idf BOW to represent sentence

• Use cosine similarity to measure distance
 

Final Extracted Summary
• Use top-ranked sentences as extracted summary

‣ Saliency (tf-idf or log likelihood ratio)

‣ Centrality

Method 4: RST Parsing
• Rhetorical structure theory (L12, Discourse): explain how clauses are connected

• Define the types of relations between a nucleus (main clause) and a satellite (supporting clause)

• Nucleus more important than satellite

• A sentence that functions as a nucleus to more sentences = more salient

Extractive: Multi-Doc
Summarisation System
• Similar to single-document extractive summarisation system

• Challenges:
‣ Redundancy in terms of information
‣ Sentence ordering

Content Selection
• We can use the same unsupervised content selection methods (tf-idf, log likelihood ratio, centrality) to select salient(突出的) sentences

• But ignore sentences that are redundant(多余的)

Maximum Marginal Relevance
• Iteratively select the best sentence to add to summary

• Sentences to be added must be novel

• Penalise a candidate sentence if it’s similar to extracted sentences:
 
• Stop when a desired number of sentences are added

Information Ordering
• Chronological ordering:
‣ Order by document dates

• Coherence:
‣ Order in a way that makes adjacent sentences similar
‣ Order based on how entities are organised (centering theory, L12)

Sentence Realisation
• Make sure entities are referred coherently
‣ Full name at first mention
‣ Last name at subsequent mentions

• Apply coreference methods to first extract names

• Write rules to clean up

Abstractive: Single-Doc

Example
 
• Paraphrase

• A very difficult task

• Can we train a neural network to generate summary?

Encoder-Decoder?
 
• What if we treat:

‣ Source sentence = “document”

‣ Target sentence = “summary”

Data
• News headlines

• Document: First sentence of article

• Summary: News headline/title

• Technically more like a “headline generation task”

More Summarisation Data
• But headline generation isn’t really exciting...

• Other summarisation data:

‣ CNN/Dailymail: 300K articles, summary in bullets

‣ Newsroom: 1.3M articles, summary by authors
- Diverse; 38 major publications

‣ XSum: 200K BBC articles
- Summary is more abstractive than other datasets

Improvements
• Attention mechanism

• Richer word features: POS tags, NER tags, tf-idf

• Hierarchical encoders
‣ One LSTM for words
‣ Another LSTM for sentences

Potential issues of an attention encoder-decoder summarisation system?
Has the potential to generate new details not in the source document

Unable to handle unseen words in the source document

Example picture
 
Copy Mechanism
• Generate summaries that reproduce details in the document

• Can produce out-of-vocab words in the summary by copying them in the document
‣ e.g. smergle = out of vocabulary
‣ p(smergle) = attention probability + generation probability = attention probability

Latest Development
• State-of-the-art models use transformers instead of RNNs

• Lots of pre-training

• Note: BERT not directly applicable because we need a unidirectional decoder (BERT is only an encoder)

Evaluation
ROUGE
• Similar to BLEU, evaluates the degree of word overlap between generated summary and reference/human summary

• But recall oriented

• Measures overlap in N-grams separately (e.g. from 1 to 3)

• ROUGE-2: calculates the percentage of bigrams from the reference that are in the generated summary

ROUGE-2: Example
 

Revise lecture (lecture 24)





