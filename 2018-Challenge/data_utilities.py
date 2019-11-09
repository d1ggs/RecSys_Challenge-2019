import csv
from tqdm import tqdm
from scipy.sparse import csr_matrix
import numpy as np

def build_URM(filename):
    URM_tuples = []

    with open(filename) as csvinput:
        inputreader = csv.reader(csvinput, delimiter=',')
        for csv_row in inputreader:
            URM_tuples.append(tuple(csv_row))
        csvinput.close()

    # Build COO representation
    rows, cols = [], []

    URM_tuples = URM_tuples[1:]

    for URM_tuple in URM_tuples:
        rows.append(int(URM_tuple[0]))
        cols.append(int(URM_tuple[1]))

    ones = np.array([1]*len(URM_tuples)) # Assign 1 where there has been an interaction

    # Obtain CSR representation
    
    URM_csr = csr_matrix( (ones, (np.array(rows), np.array(cols)) ) )

    return URM_csr # First row contains columns headers, not useful

def build_ICM(filename, tag_name):
    ICM_tuples = []

    with open(filename) as csvinput:
        inputreader = csv.reader(csvinput, delimiter=',')
        for csv_row in inputreader:
            ICM_tuples.append(tuple(csv_row))
        csvinput.close()
    
    tag_index = 0
    for i in range(len(ICM_tuples[0])):
        if ICM_tuples[0][i] == tag_name:
            tag_index = i
            break

    print("Parameter " + tag_name + " has index " + str(tag_index))
    ICM_tuples = ICM_tuples[1:] # Remove header row

    rows = []
    cols = []
    for ICM_tuple in ICM_tuples:
        rows.append(int(ICM_tuple[0]))
        cols.append(int(ICM_tuple[tag_index]))
    
    ones = np.array([1]*len(ICM_tuples))

    ICM = csr_matrix( (ones, (np.array(rows), np.array(cols)) ) )

    return ICM


def write_submission(recommender, target_users):
    with open("submission.csv", "w+") as submission:
        submission.write("playlist_id,track_ids\n")

        for user in tqdm(target_users):
            #print(playlist)
            line = user + ','
            for pred in recommender.recommend(int(user)):
                line = line + str(pred) + ' '
            line = line[:-1] + '\n'
            submission.write(line)

        submission.close()

def get_targets(filename):
    target_ids = []
    
    with open(filename) as target:
        target_reader = csv.reader(target)
        for row in target_reader:
            target_ids.append(row[0])
        target.close()
        
    return target_ids[1:]