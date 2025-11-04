import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VibeVerse",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. Custom CSS ---
def local_css():
    st.markdown("""
        <style>
        /* --- Main App Background --- */
        .stApp {
            background-image: url('https://i.pinimg.com/1200x/be/5d/d1/be5dd1e6535566c9784a2275a0e8d46c.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            font-family: 'Arial', sans-serif;
        }

        /* --- Text Visibility Fix --- */
        h1, h3, p, label, .stRadio {
            color: #333333 !important; 
        }

        /* --- Subtitle Size Increase --- */
        h3 {
            font-size: 1.5em !important;
            font-weight: 500 !important;
        }

        /* --- Title --- */
        h1 {
            color: #4a4a4a !important;
        }
        
        /* --- Recommendation Cards --- */
        [data-testid="stContainer"] {
            border: 1px solid #d3d3d3; 
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.95); 
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
        }

        /* --- Button --- */
        .stButton button {
            background: linear-gradient(to right, #6a11cb, #2575fc);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton button:hover {
            background: linear-gradient(to right, #5e0fb3, #1e66db);
            color: white;
        }
        
        </style>
        """, unsafe_allow_html=True)

local_css()

# --- 3. Load Saved Model Assets ---

# --- Build Absolute Paths to Files ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_TFIDF = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
FILE_COSINE = os.path.join(BASE_DIR, 'cosine_similarity.npy')
FILE_DF = os.path.join(BASE_DIR, 'main_dataframe.pkl')
FILE_INDICES = os.path.join(BASE_DIR, 'indices.pkl')

@st.cache_data
def load_data():
    """Loads all the pre-computed model files."""
    try:
        tfidf = joblib.load(FILE_TFIDF)
        cosine_sim = np.load(FILE_COSINE)
        df = pd.read_pickle(FILE_DF)
        indices = pd.read_pickle(FILE_INDICES)
        return tfidf, cosine_sim, df, indices
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found. The app is looking for a file that isn't in your GitHub repository. Please double-check all 4 model file names. Missing file: {e.filename}")
        return None, None, None, None

tfidf, cosine_sim, df, indices = load_data()

# --- 4. The Recommendation Function (FIXED!) ---
def get_recommendations(title, target_type='all', top_n=5):
    """
    Finds the most similar items based on title and target type.
    """
    if title not in indices:
        return pd.DataFrame(columns=['title', 'creator', 'type', 'similarity'])

    idx = indices[title]
    
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_items = []
    for i, score in sim_scores:
        if i == idx:
            continue
        item_type = df.iloc[i]['type']
        
        # --- THIS IS THE FIX ---
        # We lowered the threshold from 0.01 to 0.0 (Show ANY match)
        # This will show even the weakest 0.1% matches (which show as 0%)
        if (target_type == 'all' or item_type == target_type) and score >= 0: # Changed from > 0.01
            recommended_items.append({
                'title': df.iloc[i]['title'],
                'creator': df.iloc[i]['creator'].title(),
                'type': item_type,
                'similarity': score
            })

        if len(recommended_items) >= top_n:
            break
            
    return pd.DataFrame(recommended_items)

# --- 5. Helper Function for Display ---
def display_recommendation(row):
    """Creates a single recommendation card."""
    icon = "📚" if row['type'] == 'book' else "🎧"
    with st.container(border=True):
        st.write(f"**{icon} {row['title']}** by {row['creator']}")
        # We will show 0% for very weak matches, which is fine
        st.progress(float(row['similarity']), text=f"Match: {row['similarity']:.0%}")
    st.write("")

# --- 6. Build the Streamlit User Interface (UI) ---
if df is not None:
    
    st.title("🎧 VibeVerse 📚")
    st.subheader("Find a book that matches your favorite song, or a song that matches your favorite book.")
    st.divider() 

    # Prepare the separate title lists
    book_titles = sorted(df[df['type'] == 'book']['title'].unique())
    song_titles = sorted(df[df['type'] == 'song']['title'].unique())

    # Create two columns for the inputs
    in_col1, in_col2 = st.columns(2)
    
    with in_col1:
        selected_book = st.selectbox(
            "Choose a book...",
            options=book_titles,
            placeholder="Search for a book",
            index=None 
        )
    
    with in_col2:
        selected_song = st.selectbox(
            "Choose a song...",
            options=song_titles,
            placeholder="Search for a song",
            index=None 
        )
    
    st.write("") 
    
    target_type = st.radio(
        "What do you want recommendations for?",
        ('All', 'Book', 'Song'),
        horizontal=True,
        index=0 
    )

    if st.button("Find My Vibe"):
        
        selected_title = None
        if selected_book and selected_song:
            st.warning("Please choose only ONE title (a book OR a song). Clear the other selection to continue.")
        elif selected_book:
            selected_title = selected_book
        elif selected_song:
            selected_title = selected_song
        else:
            st.warning("Please select a title first!")

        if selected_title:
            if selected_title not in indices:
                st.error(f"Title '{selected_title}' not found. Please check spelling or try another title.")
            else:
                try:
                    input_item_type = df.iloc[indices[selected_title]]['type'].capitalize()
                    st.header(f"Because you like the {input_item_type} '{selected_title}'...")
                except Exception:
                     st.header(f"Because you like '{selected_title}'...")
                st.write("") 

                target = target_type.lower()
                
                if target == 'all':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📚 Book Matches")
                        book_recs = get_recommendations(selected_title, target_type='book')
                        if book_recs.empty:
                            st.write("No matching books found.")
                        else:
                            for _, row in book_recs.iterrows():
                                display_recommendation(row)
                    with col2:
                        st.subheader("🎧 Song Matches")
                        song_recs = get_recommendations(selected_title, target_type='song')
                        if song_recs.empty:
                            st.write("No matching songs found.")
                        else:
                            for _, row in song_recs.iterrows():
                                display_recommendation(row)
                                
                elif target == 'book':
                    st.subheader("📚 Book Matches")
                    book_recs = get_recommendations(selected_title, target_type='book')
                    if book_recs.empty:
                        st.write("No matching books found.")
                    else:
                        for _, row in book_recs.iterrows():
                            display_recommendation(row)
                            
                elif target == 'song':
                    st.subheader("🎧 Song Matches")
                    song_recs = get_recommendations(selected_title, target_type='song')
                    if song_recs.empty:
                        st.write("No matching songs found.")
                    else:
                        for _, row in song_recs.iterrows():
                            display_recommendation(row)
