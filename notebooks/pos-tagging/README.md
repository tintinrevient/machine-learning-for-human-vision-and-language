## POS (Part Of Speech) Tagging

### Criteria

* **semantic** criteria: what does the word refer to?
* **distributional** criteria: in which context can the word occur?
* **formal** criteria: what form does the word have?

![criteria](./pix/criteria.png)

distributional and formal criteria are useful to navigate the "unknown" words.

![unknown-words](./pix/unknown-words.png)

### Ambiguity

* 10.4% **word types** have 2-7 POS tags.
* above 50% **word tokens** are ambiguous.

<p float="left">
	<img src="./pix/ambiguity-1.png" width="400" />
	<img src="./pix/ambiguity-2.png" width="400" />
</p>

ambiguity in different languages:
![ambiguity-3](./pix/ambiguity-3.png)

### Strategy

* **uni-gram tagging**: assign to each word the most common tag (90% accuracy). 
* **bi-gram tagging**: assign to each word the most likely tag given the preceding tag.
	* cascading wrong tags
	* no lookahead

![bi-gram-tagging](./pix/bi-gram-tagging.png)

### Bayes' Theorem

<p float="left">
	<img src="./pix/bayes-theorem-equation.png" width="223" />
</p>

* P(A|B) is a conditional probability: the likelihood of event A occurring given that event B is true.
* P(B|A) is a conditional probability: the likelihood of event B occurring given that event A is true.
* P(A) and P(B) are the probabilities: the likelihood of event A and event B occurring independently of each other.

<p float="left">
	<img src="./pix/bayes-theorem.png" width="600" />
</p>

### Computation

* a sentence w<sub>1</sub><sup>n</sup> is a sequence of n words: (w1 w2 ... wn).
* POS tags t<sub>1</sub><sup>n</sup> for the sentence is a sequence of n tags: (t1 t2 ... tn).

**goal** - **most probable** tag sequence:

<p float="left">
	<img src="./pix/most-probable-tag-sequence.png" width="380" />
</p>

**step 1** - use Bayes' theorem:

<p float="left">
	<img src="./pix/using-bayes-theorem.png" width="785" />
</p>

**step 2** - ignore the denominator:

* P(t<sub>1</sub><<sup>n</sup>): **prior term**.
* P(w<sub>1</sub><<sup>n</sup>|t<sub>1</sub><<sup>n</sup>): **likelihood term**.

<p float="left">
	<img src="./pix/ignore-denominator.png" width="490" />
</p>

**step 3** - simplify assumptions:

for **prior term**, use bi-gram of tags.

<p float="left">
	<img src="./pix/prior-term.png" width="347" />
</p>

for **likelihood term**, the probability of a word only depends on its POS tag, not on other words/tags in the sequence.

<p float="left">
	<img src="./pix/likelihood-term.png" width="410" />
</p>

**model** - equation:

<p float="left">
	<img src="./pix/model.png" width="555" />
</p>

### Estimate the parameters of the model from a corpus

**tag transition probabilities**

<p float="left">
	<img src="./pix/tag-transition-probabilities.png" width="442" />
</p>

example from Brown corpus:
![brown-corpus-example-1](./pix/brown-corpus-example-1.png)

**word likelihood probabilities**

<p float="left">
	<img src="./pix/word-likelihood-probabilities.png" width="402" />
</p>

example from Brown corpus:
![brown-corpus-example-2](./pix/brown-corpus-example-2.png)

### HMM (Hidden Markov Model)

elements of HMM (**probabilistic finite state machine**):
* a set of N states/tags (N tags)
* an output of V words (V words)
* initial state (beginning of a sentence)
* **transition probability**: 
	* P(ti|ti−1)
	* A is an N x N matrix of transition probability, aij is the probability of transitioning from state i to state j.
* **emission probability**: 
	* P(wi|ti)
	* B is an N x V matrix of emission probability, bi(o) is the probability of emitting o from state i.

**λ = (A, B) is the parameter of a HMM**.

the equation in a HMM:
* output sequence O = (o1 o2 ... oT)
* tag sequence Q = (q1 q2 ... qT)

<p float="left">
	<img src="./pix/hmm.png" width="510" />
</p>

### Viterbi Algorithm

**enumeration of tag sequence won't work**
* c = 10 possible tags, n = 10 words, enumerated tag sequences = c<sup>n</sup> = 10,000,000,000 tag sequences

**Viterbi intuition**
* the best path of length t ending in state j must include the best path of length t−1 at the previous state i.
* v(j, t) is the probability of the best word sequence (o1 o2 ... ot) that ends in state j.

<p float="left">
	<img src="./pix/viterbi.png" width="580" />
</p>

example of Viterbi
* matrix of transition probability A: 2 x 2, N = {q1, q2}
* matrix of emission probability B: 2 x 3, V = {x, y, z}
* the word sequence (o1 o2 o3), that is (x z y)
* compute the tag sequence? answer is (q1 q1 q2)

![viterbi-example-1](./pix/viterbi-example-1.png)
![viterbi-example-2](./pix/viterbi-example-2.png)
![viterbi-example-3](./pix/viterbi-example-3.png)
