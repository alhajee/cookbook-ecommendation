# cookbook-recommendation

## Product recommendation system using collaborative filtering using a portion of Amazaon dataset

## Dataset & description can be downloaded from Stanford University https://snap.stanford.edu/data/web-Amazon.html
[Datasets are not included due to large filesize]

For this project, we download two files "meta_Books.json.gz", "Books_5.json.gz"

Afterwhich, we extract them into the "./input/Amazon/" parent as;
"./input/Amazon/meta_Books.json"
"./input/Amazon/Books_5.json"

Given the size of the parent dataset, we're going to preprocess it using our custom helper functions.
The helper functions can be found in the "./helpers/helper.py" file
