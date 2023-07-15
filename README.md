# Link Prediction on YouTube Collaboration Network #
<br>
Used YouTube collaboration graph from Koch et al. (2018).
<br><br>

**Using Similarity Measures**<br>

Measure| Edges Removed | Accuracy
:--| --: | --:
Adamic-Adar|20%| 13.09%
Jaccard|20%|5.18%
Dice|20%|5.18%
Adamic-Adar|30%| 11.71%
Jaccard|30%|5.18%
Dice|30%|5.18%

**Steps:**
1. Randomy sample $f$ percent of edges fron original graph $G$, to get $edge_{delete}$
2. Let $edge_{count} =$ count of edges in $edge_{delete}$
3. Create new graph $G'$, where $G' = G - edge_{delete}$
4. Get all possible node-pair combinations from $G'$
5. For each $G'$ node-pair, compute similarity score
6. Get $k$ node-pairs with highest similarity scores,<br>where $k =$ edge count  in original graph $G'$. Let's call these $Edge_{new}$
7. Get count of $edge_{delete}$ existing in  $Edge_{new}$. Let's call this $edge_{intersect}$
8. $$Accuracy = \frac{edge_{intersect}} {edge_{count}}$$

Above steps are guided by lecture slides on Predictive Models
<br>
<br>
**Using GraphSage**<br>

|Feature Type| Edges Removed | Test Accuracy|
:--| --: | --:
Node centrality<br>attributes only|20%| 83.96%
Normalized channel<br>attributes only|20%| 71.01%
**Node centrality attributes <br>with Community Dummies**|**20%**|**85.25%**
Node and normalized <br>channel attributes only|20%| 84.03%
Node and Channel Attributes <br> with Community Dummies |20%| 84.96%


**GraphSage Hyperparameters** 
|Feature Type| Value |
:--| --: | 
Batch Size|20|
Epochs|50|
Samples|[20, 10]|
Layer Sizes |[20, 20]

Trained using StellarGraph library.<br>

**References:**<br>
Christian Koch, Moritz Lode, Denny Stohr, Amr Rizk, & Ralf Steinmetz. (2018). Collaborations on YouTube: From Unsupervised Detection to the Impact on Video and Channel Popularity.

**Collaborators:**<br>
Nathan Casanova - https://www.linkedin.com/in/natescasanova/ <br>
Kaushik Asok - https://www.linkedin.com/in/kaushik-asok-002b19169/<br>
Prasanna Govindarajan - https://www.linkedin.com/in/prasannagovindarajan/
Anusha Mediboina  - https://www.linkedin.com/in/anusha-mediboina/
Janita Bose - https://www.linkedin.com/in/janitabose/
