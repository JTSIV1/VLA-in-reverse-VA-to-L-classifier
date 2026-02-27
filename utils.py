import os
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import lru_cache

from config import (
    SPACY_MODEL, LANG_ANNOTATIONS_SUBDIR, LANG_ANNOTATIONS_FILE,
    IMAGE_KEY, EPISODE_TEMPLATE,
)

# Load spaCy once gracefully
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    from spacy.cli import download
    download(SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)

def extract_verb(text):
    """Extracts verbs and their particles (e.g. 'pick up') from text."""
    doc = nlp(text.lower())
    actions = []
    
    for token in doc:
        if token.pos_ == "VERB" and (token.dep_ in ("ROOT", "conj", "xcomp", "advcl")):
            parts = [t.text for t in token.children if t.dep_ == "prt"]
            full_verb = " ".join([token.text] + parts)
            actions.append(full_verb)

    return actions

@lru_cache(maxsize=2)
def load_calvin_to_dataframe(data_dir):
    """
    Reads the CALVIN auto_lang_ann.npy file and structures the
    episodes into a Pandas DataFrame for easier filtering/access.
    """
    print("Loading CALVIN annotations from", data_dir)
    lang_path = os.path.join(data_dir, LANG_ANNOTATIONS_SUBDIR, LANG_ANNOTATIONS_FILE)
    if not os.path.exists(lang_path):
        raise FileNotFoundError(f"Annotations not found at {lang_path}")
    
    lang_data = np.load(lang_path, allow_pickle=True).item()
    instructions = lang_data['language']['ann']
    indices = lang_data['info']['indx']
    
    # Create Base DataFrame
    df = pd.DataFrame({
        'start_idx': [idx[0] for idx in indices],
        'end_idx': [idx[1] for idx in indices],
        'instruction': instructions
    })
    
    # Extract verbs and add them as columns
    print("Processing language instructions with spaCy...")
    df['verbs'] = df['instruction'].apply(extract_verb)
    
    # Filter out rows that don't have exactly 1 verb
    initial_count = len(df)
    df = df[df['verbs'].apply(len) == 1].copy()
    print(f"Filtered out {initial_count - len(df)} examples that had 0 or >1 verbs.")
    
    # Use the single extracted verb as the primary label
    df['primary_verb'] = df['verbs'].apply(lambda x: x[0])
    
    return df

def visualize_frames(df, data_dir, num_samples=3):
    """
    Randomly samples trajectories from the DataFrame and plots 
    their first and last frames.
    """
    if df.empty:
        print("DataFrame is empty. Cannot visualize.")
        return

    samples = df.sample(min(num_samples, len(df)))
    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 4 * len(samples)))
    
    # Ensure axes is a 2D array even if we only request 1 sample
    if len(samples) == 1:
        axes = np.array([axes])
        
    for i, (_, row) in enumerate(samples.iterrows()):
        start_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(row['start_idx']))
        end_path = os.path.join(data_dir, EPISODE_TEMPLATE.format(row['end_idx']))

        try:
            start_img = np.load(start_path)[IMAGE_KEY]
            end_img = np.load(end_path)[IMAGE_KEY]
            
            axes[i, 0].imshow(start_img)
            axes[i, 0].set_title(f"Start | Verb: {row['primary_verb']}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(end_img)
            axes[i, 1].set_title(f"End Frame")
            axes[i, 1].axis('off')
        except FileNotFoundError:
            axes[i, 0].text(0.5, 0.5, "Image file not found", ha='center')
            axes[i, 1].text(0.5, 0.5, "Image file not found", ha='center')
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()