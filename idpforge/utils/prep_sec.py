import numpy as np
import pandas as pd
from collections import defaultdict

def parse_df(df, fragment_length):
    # Initialize a dictionary to store the fragments and their associated secondary structure annotations
    fragment_dict = defaultdict(lambda: defaultdict(int))
    
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        sequence = row['sequence']
        sec = row['sec']
        
        # Slide through the sequence with the given fragment length
        for i in range(len(sequence) - fragment_length + 1):
            fragment = sequence[i:i + fragment_length]
            annotation = sec[i:i + fragment_length]
            
            # Record the associated secondary structure annotations
            fragment_dict[fragment][annotation] += 1
    
    # Calculate the occurrence probabilities
    fragment_probabilities = {}
    for fragment, annotations in fragment_dict.items():
        total_count = sum(annotations.values())
        fragment_probabilities[fragment] = {k: v / total_count for k, v in annotations.items()}
    
    return fragment_probabilities
    
def fetch_sec_from_seq(sequence, nsamples, db_df, xmer_prob=[1, 1, 3, 3, 1]):
    # Normalize the probabilities
    xmer_prob = np.array(xmer_prob)
    xmer_prob = xmer_prob / xmer_prob.sum()
    db_dicts = [parse_df(db_df, i+1) for i in range(len(xmer_prob))]
    samples = []
    for _ in range(nsamples):
        annotate = ""
        i = 0
        while i < len(sequence):
            # Randomly select a chunk length based on the given probabilities
            chunk_length = np.random.choice(range(1, len(xmer_prob) + 1), p=xmer_prob)
            
            # Ensure the chunk length does not exceed the remaining sequence length
            chunk_length = min(chunk_length, len(sequence) - i)
            db_dict = db_dicts[chunk_length - 1]
            
            while chunk_length > 0:
                # Extract the chunk
                chunk = sequence[i: i + chunk_length]
                if chunk in db_dict:
                    # Fetch the annotation based on occurrence probabilities
                    annotate_prob = db_dict[chunk]
                    sec = np.random.choice(list(annotate_prob.keys()), 
                        p=list(annotate_prob.values()))
                    annotate += sec
                    i += chunk_length
                    break
                else:
                    # Reduce the chunk length by one if the key is not present
                    chunk_length -= 1
        samples.append(annotate)
    return samples
    

if __name__ == "__main__":
    import argparse
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence')
    parser.add_argument('database')
    parser.add_argument('nsample', type=int)
    parser.add_argument('output', type=str)
    
    args = parser.parse_args()
    with open(args.database, "rb") as f:
        sc, sq, _ = pickle.load(f)
    SEC_database = pd.DataFrame({"sec": sc, "sequence": sq})
    sec_samples = fetch_sec_from_seq(args.sequence, args.nsample, SEC_database)
    
    with open(args.output, "w") as f:
        f.write("\n".join(sec_samples))
        
