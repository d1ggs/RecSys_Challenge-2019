import csv
from tqdm import tqdm

def prepare_data(filename):
    URM_tuples = []

    with open(filename) as csvinput:
        inputreader = csv.reader(csvinput, delimiter=',')
        for csv_row in inputreader:
            URM_tuples.append(tuple(csv_row))
        csvinput.close()

    return URM_tuples[1:] # First row contains columns headers, not useful

def write_submission(recommender, target_users):
    with open("submission.csv", "w+") as submission:
        submission.write("playlist_id,track_ids\n")

        for user in tqdm(target_users):
            #print(playlist)
            line = user + ','
            for pred in recommender.predict(int(user)):
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