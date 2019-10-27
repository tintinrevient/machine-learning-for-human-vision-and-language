## Supertagging

Supertagging is based on **"strongly lexicalised"** grammar.

**CCG (Combinatory Categorial Grammar)** is a strongly lexicalised grammar.

for each word, there is:
* **lexical categories**
	* atomic categories: S, N, NP, PP, etc...
	* complex categories: built recursively from atomic categories and slashes.
		* **intransitive verb** - S\NP: walk
		* **transitive verb** - (S\NP)/NP: respect
		* **ditransitive verb** - ((S\NP)/NP)/NP: give
* **semantic interpretation**
	* λx.WALK(x): walk
* **combinators**: rules which define how lexical categories can be combined.
	* **forward application (>)**
	* **backward application (<)**
	* **type raising (T)**
	* **forward composition (B)**

<img src="./pix/rule-1.png" width="430" />
<img src="./pix/rule-2.png" width="430" />
<img src="./pix/rule-3.png" width="430" />

examples of CCG derivation
![supertagging-derivation-1](./pix/supertagging-derivation-1.png)
![supertagging-derivation-2](./pix/supertagging-derivation-2.png)
