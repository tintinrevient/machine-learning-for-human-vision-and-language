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