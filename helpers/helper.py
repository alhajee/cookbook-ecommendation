# file manipulation
import gzip
import json
import simplejson
import pandas as pd
import numpy as np

# Functions to read and save our dataframes to file
import pickle

# Progressbar
import progressbar

# Euclidean distance
from scipy import spatial


# read our pickle from file
def load_pickle(filename):
    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    return(data)
    
# for future use we will save the dict as a file
def save_pickle(data, filename):
    pickle_out = open(filename, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    
    return True


# method to parse json from zip archive
def parse_archive(filename, archive=False):
    if archive:
        f = gzip.open(filename, 'r')
    else:
        f = open(filename, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colonPos = l.find(b':') if archive else l.find(':')
        if colonPos == -1:
          yield entry
          entry = {}
          continue
        eName = l[:colonPos]
        rest = l[colonPos+2:]
        entry[eName] = rest
        yield entry


# method to parse json from file
def parse_json(filename):
    with open(filename, 'r') as f:
        for l in f:
            yield l

# selectably format a json file
def format_json(filename: str, columns: list) -> list:
    rows = list()

    for raw in parse_json(filename):
        a = json.loads(raw)

        row = dict()

        for col in columns:
            row[col] = a.get(col)
        rows.append(row)
    return rows
        

# load reviews from json file
def load_reviews(filename):
    columns = ['overall', 'reviewerID', 'asin', 'unixReviewTime']
    df = pd.DataFrame(format_json(filename=filename, columns=columns))
    return df


# load metadata from json file
def load_metadata(filename):
    columns = ['category', 'title', 'rank', 'also_buy', 'also_view', 'price', 'asin']
    df = pd.DataFrame(format_json(filename=filename, columns=columns))
    return df


# save a dataframe to file
def to_csv(df, filename: str):
    try:
        compression_opts = dict(method='zip', archive_name='out.csv')
        df.to_csv(filename, index=False)
        del df
        return filename
    except:
        return None


def clean_category(categories: list):
    """
    A simple function that parses and returns a string of categories
    ...
    
    Parameters
    ----------
    categories: list
        list of categories.
        
    Returns
    ------
     category : string
        A string containing a concatenated list of categories.
    """
    # convert list to text
    return "|".join(categories)

def extract_digits(rank: list) -> int:
    """
    A simple function that extracts digits from a string with mixed 
    characters (Numbers, Strings, Characters)
    ...
    
    Parameters
    ----------
    rank: list
        A list containing string that represent the rank of an item.
        
    Returns
    ------
    int
    """
    rank = str(rank) # convert list to string
    
    # extract numbers from text
    text = [s for s in rank if s.isdigit()]
    
    # convert list to string
    text = ''.join(text)
    
    # return 0 if blank
    text = text if len(text) > 0 else 0
    
    return int(text)

def clean_also_buy(asins: list) -> str:
    """
    A simple function that converts a list to string with delimeter
    ...
    
    Parameters
    ----------
    asins: list
        A list containing item ids.
        
    Returns
    ------
     string
    """
    # convert list to text
    return "|".join(asins)

def clean_also_view(asins: list) -> str:
    """
    A simple function that converts a list to string with delimeter
    ...
    
    Parameters
    ----------
    asins: list
        A list containing item ids.
        
    Returns
    ------
     string
    """
    # convert list to text
    return "|".join(asins)

def clean_price(price: str) -> float:
    """
    A simple function strips a string of its dollar sign
    and returns the float equivalent
    ...
    
    Parameters
    ----------
    price: string
        
    Returns
    ------
     price: float
    """
    price = price.replace('$', '').replace(',', '')
    
    # return 0 if blank
    price = price if len(price) > 0 else 0
    
    # return 0 if failed to convert to float
    try:
        price = float(price)
    except:
        price = 0
        
    return price


# clean features
def clean_metadata_df(df_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    A simple function that applies function to cleans the features
    (category, rank, also_buy, also_view, price) in df_metadata dataframe
    ...
    
    Parameters
    ----------
    df_metadata: pd.DataFrame
        dataframe that contains metadata on items.
        
    Returns
    ------
    df_reviews : pd.DataFrame
        A modified dataframe that contains metadata on items.
    """
    df_metadata['category'] = df_metadata['category'].apply(clean_category)
    df_metadata['rank'] = df_metadata['rank'].apply(extract_digits)
    df_metadata['also_buy'] = df_metadata['also_buy'].apply(clean_also_buy)
    df_metadata['also_view'] = df_metadata['also_view'].apply(clean_also_view)
    df_metadata['price'] = df_metadata['price'].apply(clean_price)

    return df_metadata


# get unique list of categories
def get_categories(categories: list):
    """
    A simple function that parses and returns the unique list of categories
    ...
    
    Parameters
    ----------
    categories: list
        list of categories.
        
    Returns
    ------
    unique : list
        unique list of categories.
    """
    unique = list()
    
    for item in categories:
        if 'Books|' in item:
            item = item.split('|')
            unique.extend(item)
        else:
            pass
        
    unique = set(unique)
    
    return list(unique)

# data engineering on reviews
def clean_reviews(df_reviews: pd.DataFrame):
    """
    A simple function that cleans (renames the previously named feature
    `overall` to `rating`) in df_reviews dataframe
    ...
    
    Parameters
    ----------
    df_reviews: pd.DataFrame
        dataframe that contains user ratings of items.
        
    Returns
    ------
    df_reviews : pd.DataFrame
        A modified dataframe that contains user ratings of items.
    """
    df_reviews = df_reviews.rename(columns={'overall':'rating'}, inplace=False)
    return df_reviews


    #  data engineering on items
def clean_items(df_items: pd.DataFrame, df_metadata: pd.DataFrame):
    """
    A simple function that creates new features average_rating, num_rating,
    title in a dataframe
    ...
    
    Parameters
    ----------
    df_items: pd.DataFrame
        dataframe that contains items.
    df_metadata: pd.DataFrame
        metadata dataframe that contains data about items.
        
    Returns
    ------
    df_items : pd.DataFrame
        A modified dataframe that contains data on items.
    """
    df_items = pd.DataFrame(df_reviews.groupby('asin')['rating'].mean())
    df_items.rename(columns={'rating': 'avg_rating'}, inplace=True)
    df_items['num_rating'] = pd.DataFrame(df_reviews.groupby('asin')['rating'].count())

    df_items = pd.merge(df_items, df_metadata[['asin', 'title']], on='asin')
    
    return df_items
    

def load_dataframes(metadata_file, reviews_file, items_file):
    """
    A simple function that reads csv files and load them into a dataframe
    ...
        
    Returns
    ------
    tuple of pd.DataFrame -> df_metadata, df_reviews, df_items
        df_metadata: dataframe containing metadata of items
        df_reviews: dataframe that contains user ratings of items
        df_items: dataframe that contain items
    """
    df_metadata = pd.read_csv(metadata_file)
    df_reviews = pd.read_csv(reviews_file)
    df_items = pd.read_csv(items_file)
    
    return (df_metadata, df_reviews, df_items)


def save_dataframes():
    """
    A simple function that saves dataframes (df_metadata, df_reviews, df_items)
    into a csv file for purpose of backup
    """
    # save metadata dataframe to csv file
    to_csv(df_metadata, "df_metadata.csv")
    
    # save reviews dataframe to csv file
    to_csv(df_reviews, "df_reviews.csv")
    
    # save items dataframe to csv file
    to_csv(df_items, "df_items.csv")


def n_times_rated_items(min_ratings: int, df_reviews, df_items, df_metadata) -> set:
    """
    A simple function that returns items in a dataframe
    which satisfy a certain constraint (number of times an item has been rated)
    ...
    
    Parameters
    ----------
    min_ratings: int
        minimum number of rating that an item has to have.
    
    df_reviews: pd.DataFrame
        dataframe that contains user ratings of items.
    
    df_items: pd.DataFrame
        dataframe that contains items.
    
    df_metadata: pd.DataFrame
        dataframe that contains metadata on items.
        
    Returns
    ------
    (df_num_ratings, df_reviews, df_items, df_metadata) : set
        Modified dataframes that satisfy the constraint.
    """
    # Users that rated up to n items
    df_num_ratings = df_items[df_items.num_rating >= min_ratings]
    
    # deleting review of items with less than n reviews
    df_reviews = df_reviews[df_reviews['asin'].isin(df_num_ratings['asin'])]
    # deleting items that are not in new reviews
    df_items = df_items[df_items['asin'].isin(df_reviews['asin'].unique())]
    # delete metadata of items that are not in new items dataset
    df_metadata = df_metadata[df_metadata['asin'].isin(df_items['asin'])]

    # there are reviews of items that do not have metadata
    # - delete such reviews
    df_reviews = df_reviews[df_reviews['asin'].isin(df_num_ratings['asin'])]
    
    return (df_num_ratings, df_reviews, df_items, df_metadata)


def n_times_rated_users(df_num_user_ratings, min_ratings: int, df_reviews, df_items, df_metadata) -> set:
    """
    A simple function that returns the items in a dataframe
    which satisfy a certain constraint (number of times a user has rated items)
    ...
    
    Parameters
    ----------
    df_num_user_rating: pd.DataFrame
        dataframe that contains the number of times a user has rated items
    
    min_ratings: int
        minimum number of rating that an item has to have.
    
    df_reviews: pd.DataFrame
        dataframe that contains user ratings of items.
    
    df_items: pd.DataFrame
        dataframe that contains items.
    
    df_metadata: pd.DataFrame
        dataframe that contains metadata on items.
        
    Returns
    ------
    (df_num_ratings, df_reviews, df_items, df_metadata) : set
        Modified dataframes that satisfy the constraint.
    """
    # Users that rated up to n items
    df_num_user_ratings = df_num_user_ratings[df_num_user_ratings.num_ratings >= min_ratings]
    
    # deleting review of items with less than n reviews
    df_reviews = df_reviews[df_reviews['reviewerID'].isin(df_num_user_ratings['reviewerID'])]
    # deleting items that are not in new reviews
    df_items = df_items[df_items['asin'].isin(df_reviews['asin'])]
    # delete metadata of items that are not in new items dataset
    df_metadata = df_metadata[df_metadata['asin'].isin(df_items['asin'])]

    # there are reviews of items that do not have metadata
    # - delete such reviews
    df_reviews = df_reviews[df_reviews['asin'].isin(df_items['asin'])]
    
    return (df_num_user_ratings, df_reviews, df_items, df_metadata)


def get_num_user_ratings(reviews: pd.DataFrame):
    """
    A simple function that creates features to stores
    the number of items a user has rated
    ...
    
    Parameters
    ----------
    reviews: pd.DataFrame
        dataframe that contains user ratings of items.
        
    Returns
    ------
    num_user_ratings : pd.DataFrame
        A modified dataframe that contains a unique list of
        users and the number of times they rated items.
    """
    # group dataframe by reviewer and calculate the number of rating for each reviewer
    num_user_ratings = pd.DataFrame(reviews.groupby('reviewerID')['rating'].count())
    # rename the new column containing count of ratings per user
    num_user_ratings = num_user_ratings.rename(columns={'rating': 'num_ratings'})
    # reset the index in the dataframe
    num_user_ratings = num_user_ratings.reset_index()
    
    return num_user_ratings


def train_test_split(ratings: pd.DataFrame, seed: int = 1, test_size: float = 0.20):
    """
    A simple function that splits a User Dataframe into test/train based on a test size
    ...
    
    Parameters
    ----------
    ratings: pd.DataFrame
        user ratings of items.
    seed: int
        seed for random number generator that's needed for reproducibility
        default value as 1.
    test_size: float
        percentage of dataframe to store in test.
        default value as 0.20 (20%)
    
    Raises
    ------
    AssertionError
        If size of test and train dataframe do not sum up
        to the size of original dataframe raise error.
            
    Returns
    ------
    tuple -> test: pd.DataFrame, train: pd.DataFrame
        A tuple of test & train dataframes.
    """
    # initialize an empty dataframe
    test = pd.DataFrame()
    # make a copy of the dataframe
    train = ratings.copy()    
    
    # configure progress bar
    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    counter = 0
    bar = progressbar.ProgressBar(maxval=ratings['reviewerID'].nunique()+1, widgets=widgets)
    bar.start()
    
    # loop through all the users in the dataframe
    for user in ratings['reviewerID'].unique():
        # choose test ratings at random from a users's ratings
        test_ratings = ratings[ratings.reviewerID == user].sample(frac=test_size, replace=False, random_state=seed)
        # append to test sample
        test = test.append(test_ratings)
        
        # update progress bar
        counter += 1
        bar.update(counter)
    
    bar.finish() # notify progress bar that task is done
    
    # delete test ratings from train
    train = train[~train.isin(test)].dropna()
    
    # Test and training are truly disjoint
    assert(train.shape[0] + test.shape[0] == ratings.shape[0])
    
    return train, test


# returns the percentage sparsity in a matrix
def get_sparsity(matrix, values):
    total_values = matrix.shape[0] * matrix.shape[1]
    sparsity = (values.shape[0]) / total_values
    return (sparsity*100)


# Function to transforming a euclidean distance into a probablity distribution
# Added 1 because perfect similarity can make the euclidean distance zero and give NotDefined Error
def similarity(dist):
    return 1/ (1 + dist)


# Function to get reviews from the common items
def get_reviews(user_i, user_j, sim_items):
    return [(reviewer_by_item[user_i, item], reviewer_by_item[user_j, item]) for item in sim_items]


def compute_correlation_dist(user_i, user_j, items_seen, reviewer_by_item):
    """
    INPUT:
    [int] user_i - the user id of an individual
    [int] user_j - the user id of an individual
    
    OUTPUT:
    [float] dist - the euclidean distance between user_i and user_j
    """
    # items rated by user_i
    items_i = [] if items_seen.get(user_i) is None else items_seen.get(user_i)
    # items rated by user_j
    items_j = [] if items_seen.get(user_j) is None else items_seen.get(user_j)
    
    
    # items both users have seen
    sim_items = np.intersect1d(items_i, items_j, assume_unique=True)

    # there are no similar items between the users
    if sim_items.size is 0:
        return 0.0
    
    # pull the locatin of the data from our sparse dataframe
    df = reviewer_by_item.loc[(user_i, user_j), sim_items]
    # compute correlation
    
    dist = spatial.distance.euclidean(df.iloc[0], df.iloc[1])
    
    return similarity(dist) # return the euclidean distance


    # Create a dictionary of users and corresponding items seen



def get_items_seen(reviewer_by_item, reviewerID):
    '''
    INPUT:
    [int] reviewerID - the user id of an individual
    OUTPUT:
    [list] items - an array of items that user has watched
    '''
    items = reviewer_by_item.loc[reviewerID][reviewer_by_item.loc[reviewerID] > 0].index.values
     
    return items

def create_reviewer_item_dict(reviewer_by_item, df_reviews):
    '''
    INPUT: None
    OUTPUT:
    [dict] items_read - a dictionary where each key is a reviewerID and the value is an array of item_id's
    '''
    
    # configure progressbar
    items_seen = dict()
    n_users = reviewer_by_item.shape[0]
    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    counter = 0
    bar = progressbar.ProgressBar(maxval=n_users+1, widgets=widgets)
    bar.start()
    
    grouped = df_reviews.groupby('reviewerID')
    for user, group in grouped:
        items_seen[user] = get_items_seen(reviewer_by_item, user)
        counter += 1
        bar.update(counter)
   
    # end progressbar    
    bar.finish()
    
    return items_seen


# funtion to create a user by user similarity matrix
def compute_user_sim_matrix(user_matrix: pd.DataFrame, items_seen, reviewer_by_item):
    users = user_matrix.index
    num_users = len(user_matrix.index)
    
    # progressbar widget
    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    counter = 0
    bar = progressbar.ProgressBar(maxval=num_users+1, widgets=widgets)
    bar.start()
    
    
    # loop through all the users
    for user_i in users:
        
        # update progressbar
        counter += 1
        bar.update(counter)
        
        for user_j in users:
            # compute user similarity
            user_matrix.loc[user_i, user_j] = compute_correlation_dist(user_i, user_j, items_seen, reviewer_by_item)
    
    # end progressbar
    bar.finish()
    
    return user_matrix


# Function to give recommendation to users based on their reviews.
def recommend_items(user_i, users, num_suggestions):

    similarity_scores = [(compute_correlation_dist(user_i, user_j), user_j) for user_j in users if user_j != user_i]
    # Get similarity Scores for all the users
    similarity_scores.sort() 
    similarity_scores.reverse()

    recommendations = {}
    # Dictionary to store recommendations
    for similarity, user_j in similarity_scores:
        reviewed = df_reviews[df_reviews.reviewerID == user_j]
        # Storing the review
        for item in items_seen.get(user_j):
            if item not in items_seen.get(user_i):
                review = reviewer_by_item.loc[user_j, item]
                weight = similarity * review
                # Weighing similarity with review
                if item in recommendations:
                    sim, weights = recommendations[item]
                    recommendations[item] = (sim + similarity, weights + [weight])
                    # Similarity of item along with weight
                else:
                    recommendations[item] = (similarity, [weight])
                    
    for recommendation in recommendations:
        similarity, item = recommendations[recommendation]
        recommendations[recommendation] = sum(item) / similarity
        # Normalizing weights with similarity

    sorted_recommendations = sorted(recommendations, key=recommendations.__getitem__, reverse=True)
    # Sorting recommendations with weight 🤘
    return sorted_recommendations[:num_suggestions]


# function that predicts a users rating for a particular item 💦
def predict_user_rating(user_id, item_id, user_sim_matrix, reviewer_by_item):
    
    users = user_sim_matrix.index    
    predicted_rating = 0
    total_sim = 0.1                                      
    
    for user_j in users:
        # get similarity between our user and another user_j 💋
        sim = user_sim_matrix.loc[user_id, user_j]
        
        # get rating of item by user_j
        rating_j = reviewer_by_item.loc[user_j, item_id]
        
        # skip if it's the same user 😁
        if user_j == user_id:
            continue
        
        # skip if user hasn't rated the item 😉
        if rating_j == 0.0:
            continue
            
        # multiply user_j's rating of item by his corresponding similarity to our user 👨🏋️‍♀️
        predicted_rating += (sim * rating_j)
        # compute total user similarity between our user and other users 👫
        total_sim += abs(sim)
    # normalize predicted rating by dividing it by total similarity between other users 🔥
    predicted_rating /= total_sim
    return predicted_rating