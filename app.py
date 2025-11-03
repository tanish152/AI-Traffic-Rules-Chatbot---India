import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="AI Traffic Rules Chatbot - India",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme state
if 'theme' not in st.session_state:
    st.session_state.theme = 'day'  # Default to night mode

# Theme toggle function
def toggle_theme():
    st.session_state.theme = 'day' if st.session_state.theme == 'night' else 'night'

# Define color schemes
THEMES = {
    'night': {
    'bg_gradient': 'linear-gradient(135deg, #0a0f13 0%, #0c1b1e 100%)',
    'primary': '#00e0b8',  # bright teal accent
    'secondary': '#05b295',  # muted teal
    'accent': '#1ef7c6',  # glowing neon accent
    'text_primary': "#ffffff",
    'text_secondary': "#196e31",  # light desaturated teal-gray
    'card_bg': '#0f2124',  # deep dark bluish-green
    'card_border': '#00e0b8',  # teal edge highlight
    'header_gradient': 'linear-gradient(135deg, #00e0b8 0%, #027e68 100%)',
    'button_gradient': 'linear-gradient(135deg, #00e0b8 0%, #029b7c 100%)',
    'input_bg': '#071416',
    'input_border': '#027e68',
    'shadow_color': 'rgba(0, 224, 184, 0.3)',
    'sidebar_bg': 'linear-gradient(180deg, #0a0f13 0%, #0c1b1e 100%)',
},

    'day': {
        'bg_gradient': 'linear-gradient(135deg, #e7dfdc 0%, #f5f0ee 100%)',
        'primary': '#64113f',
        'secondary': '#a6788e',
        'accent': '#e7dfdc',
        'text_primary': '#2d1b2e',
        'text_secondary': '#64113f',
        'card_bg': '#ffffff',
        'card_border': '#e7dfdc',
        'header_gradient': 'linear-gradient(135deg, #64113f 0%, #a6788e 100%)',
        'button_gradient': 'linear-gradient(135deg, #64113f 0%, #a6788e 100%)',
        'input_bg': '#ffffff',
        'input_border': '#e7dfdc',
        'shadow_color': 'rgba(100, 17, 63, 0.3)',
        'sidebar_bg': 'linear-gradient(180deg, #f5f0ee 0%, #e7dfdc 100%)',
    }
}

current_theme = THEMES[st.session_state.theme]

# Custom CSS with Dynamic Theme
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    .stApp {{
        background: {current_theme['bg_gradient']};
        transition: all 0.5s ease;
    }}
    
    /* Theme Toggle Button - Fixed Position */
    .theme-toggle {{
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 9999;
        background: {current_theme['card_bg']};
        border: 2px solid {current_theme['card_border']};
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 15px {current_theme['shadow_color']};
        transition: all 0.3s ease;
        font-size: 24px;
    }}
    
    .theme-toggle:hover {{
        transform: scale(1.1) rotate(15deg);
        box-shadow: 0 6px 20px {current_theme['shadow_color']};
    }}
    
    /* Hide code blocks */
    .stMarkdown code {{
        display: none !important;
    }}
    
    pre {{
        display: none !important;
    }}
    
    /* Main Header - Centered */
    .main-header {{
        background: {current_theme['header_gradient']};
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px {current_theme['shadow_color']};
        text-align: center;
        position: relative;
        overflow: hidden;
    }}
    
    .main-header::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }}
    
    @keyframes rotate {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    .main-title {{
        color: {current_theme['text_primary']};
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .main-subtitle {{
        color: {current_theme['accent']};
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }}
    
    /* Chat Messages */
    .user-message {{
        background: {current_theme['button_gradient']};
        color: {current_theme['text_primary']};
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem auto;
        max-width: 75%;
        box-shadow: 0 5px 15px {current_theme['shadow_color']};
        animation: slideInRight 0.4s ease;
        transform: translateZ(0);
        transition: transform 0.3s ease;
    }}
    
    .user-message:hover {{
        transform: translateY(-2px) scale(1.01);
    }}
    
    @keyframes slideInRight {{
        from {{
            opacity: 0;
            transform: translateX(30px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    .bot-message {{
        background: {current_theme['card_bg']};
        color: {current_theme['text_primary']};
        padding: 1.2rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem auto 1rem 0;
        max-width: 75%;
        border: 2px solid {current_theme['card_border']};
        box-shadow: 0 5px 15px {current_theme['shadow_color']};
        animation: slideInLeft 0.4s ease;
        transform: translateZ(0);
        transition: transform 0.3s ease;
    }}
    
    .bot-message:hover {{
        transform: translateY(-2px) scale(1.01);
    }}
    
    @keyframes slideInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    /* Rule Cards with 3D Effect */
    .rule-card {{
        background: {current_theme['card_bg']};
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 2px solid {current_theme['card_border']};
        box-shadow: 0 8px 25px {current_theme['shadow_color']};
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        transform-style: preserve-3d;
        position: relative;
    }}
    
    .rule-card:hover {{
        transform: translateY(-8px) rotateX(2deg);
        box-shadow: 0 15px 40px {current_theme['shadow_color']};
        border-color: {current_theme['secondary']};
    }}
    
    .rule-title {{
        color: {current_theme['text_secondary']};
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }}
    
    .info-box {{
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        font-weight: 600;
        border: 2px solid;
        transition: all 0.3s ease;
        transform: translateZ(20px);
    }}
    
    .info-box:hover {{
        transform: translateZ(30px) scale(1.05);
    }}
    
    .fine-box {{
        background: linear-gradient(135deg, #fff5f5 0%, #ffe5e5 100%);
        color: #c41e3a;
        border-color: #e74c3c;
    }}
    
    .points-box {{
        background: linear-gradient(135deg, #fff8f0 0%, #ffe8d5 100%);
        color: #d68910;
        border-color: #f39c12;
    }}
    
    .region-box {{
        background: linear-gradient(135deg, #f0f5ff 0%, #e5efff 100%);
        color: #2563eb;
        border-color: #3b82f6;
    }}
    
    .tip-box {{
        background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
        color: #15803d;
        padding: 1.2rem;
        border-radius: 15px;
        border: 2px solid #22c55e;
        margin-top: 1.2rem;
    }}
    
    /* Filter Section */
    .filter-section {{
        background: {current_theme['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 20px {current_theme['shadow_color']};
        border: 2px solid {current_theme['card_border']};
    }}
    
    .filter-label {{
        color: {current_theme['text_secondary']};
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        display: block;
    }}
    
    /* Statistics Cards */
    .stat-card {{
        background: {current_theme['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
        border: 2px solid {current_theme['card_border']};
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px {current_theme['shadow_color']};
    }}
    
    .stat-card:hover {{
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 8px 25px {current_theme['shadow_color']};
        border-color: {current_theme['secondary']};
    }}
    
    .stat-number {{
        font-size: 2.5rem;
        font-weight: 800;
        background: {current_theme['button_gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .stat-label {{
        color: {current_theme['text_secondary']};
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: {current_theme['button_gradient']};
        color: {current_theme['text_primary']};
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 5px 15px {current_theme['shadow_color']};
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px {current_theme['shadow_color']};
    }}
    
    /* Input Fields */
    .stTextInput>div>div>input {{
        background: {current_theme['input_bg']};
        border: 2px solid {current_theme['input_border']};
        border-radius: 12px;
        color: {current_theme['text_primary']};
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: {current_theme['secondary']};
        box-shadow: 0 0 0 3px {current_theme['shadow_color']};
        outline: none;
    }}
    
    .stSelectbox>div>div {{
        background: {current_theme['input_bg']};
        border: 2px solid {current_theme['input_border']};
        border-radius: 12px;
        color: {current_theme['text_primary']};
    }}
    
    /* Confidence Badge */
    .confidence-badge {{
        display: inline-block;
        background: {current_theme['button_gradient']};
        color: {current_theme['text_primary']};
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 800;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px {current_theme['shadow_color']};
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
    }}
    
    /* Warning Box */
    .warning-box {{
        background: linear-gradient(135deg, #fff9e6 0%, #ffedd5 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #92400e;
        font-weight: 600;
    }}
    
    /* Sidebar */
    div[data-testid="stSidebar"] {{
        background: {current_theme['sidebar_bg']};
        border-right: 2px solid {current_theme['secondary']};
    }}
    
    div[data-testid="stSidebar"] .stMarkdown {{
        color: {current_theme['text_primary']};
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {current_theme['accent']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {current_theme['button_gradient']};
        border-radius: 10px;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {current_theme['text_secondary']} !important;
    }}
    
    .stMarkdown p {{
        color: {current_theme['text_primary']};
    }}
    
    /* Legal Reference Highlight */
    .legal-highlight {{
        background: linear-gradient(135deg, #fff9e6 0%, #ffedd5 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 600;
        color: #92400e;
    }}
    
    /* Input Container */
    .input-container {{
        background: {current_theme['card_bg']};
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 5px 20px {current_theme['shadow_color']};
        border: 2px solid {current_theme['card_border']};
        margin-top: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Theme Toggle Button (Fixed Position)
theme_icon = "☀️" if st.session_state.theme == 'night' else "🌙"
st.markdown(f"""
<div class="theme-toggle" onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'toggle'}}, '*')">
    {theme_icon}
</div>
""", unsafe_allow_html=True)

# Create a hidden button to handle the toggle
col_toggle = st.columns([10, 1])[1]
with col_toggle:
    if st.button(theme_icon, key="theme_toggle_btn", help="Toggle Day/Night Mode"):
        toggle_theme()
        st.rerun()

# Load and prepare data
@st.cache_data
def load_data():
    """Load the traffic rules dataset"""
    try:
        df = pd.read_excel('dataset.xlsx.xlsx', engine='openpyxl')
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file not found! Please ensure 'indian_traffic_rules_dataset_new1.csv.xlsx' is in the directory.")
        return pd.DataFrame()

# Initialize TF-IDF for semantic search
@st.cache_resource
def initialize_search_engine(df):
    """Initialize TF-IDF vectorizer for semantic search"""
    if df.empty:
        return None, None
    
    df['search_text'] = (
        df['rule_title'].fillna('') + ' ' +
        df['paraphrase'].fillna('') + ' ' +
        df['offense_type'].fillna('') + ' ' +
        df['region'].fillna('') + ' ' +
        df['vehicle_type'].fillna('')
    )
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])
    
    return vectorizer, tfidf_matrix

def extract_city_from_query(query, df):
    """Extract city name from user query"""
    query_lower = query.lower()
    cities = df['region'].unique()
    
    for city in cities:
        if city.lower() in query_lower:
            return city
    return None

def get_valid_offenses(vehicle_type, all_offenses):
    """Return valid offense types based on vehicle selection"""
    if vehicle_type == 'All':
        return all_offenses
    
    two_wheeler_vehicles = ['Motorcycle', 'Bike', 'Two-Wheeler', 'Scooter']
    four_wheeler_vehicles = ['Car', 'Truck', 'Bus', 'Four-Wheeler']
    
    helmet_offenses = ['No Helmet', 'Helmet Violation']
    seatbelt_offenses = ['No Seatbelt', 'Seatbelt Violation']
    
    if vehicle_type in two_wheeler_vehicles:
        return [o for o in all_offenses if not any(sb in o for sb in seatbelt_offenses)]
    elif vehicle_type in four_wheeler_vehicles:
        return [o for o in all_offenses if not any(h in o for h in helmet_offenses)]
    else:
        return all_offenses

def deduplicate_results(results_df):
    """Remove duplicate rules"""
    if results_df.empty:
        return results_df
    
    results_df = results_df.sort_values('similarity', ascending=False)
    deduped = results_df.drop_duplicates(
        subset=['offense_type', 'region'],
        keep='first'
    )
    
    return deduped.reset_index(drop=True)

def process_query(query, df, vectorizer, tfidf_matrix, top_k=20):
    """Process user query and return relevant rules"""
    if df.empty or vectorizer is None:
        return pd.DataFrame(), "I couldn't process your query. Please check if the dataset is loaded."
    
    mentioned_city = extract_city_from_query(query, df)
    query_vec = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    if mentioned_city:
        city_results = results[results['region'] == mentioned_city].copy()
        if not city_results.empty:
            results = city_results
    
    results = deduplicate_results(results)
    results = results.head(5)
    
    if results.empty or results['similarity'].iloc[0] < 0.1:
        response = "I couldn't find specific information about that. Could you rephrase your question?"
    else:
        if mentioned_city:
            response = f"""**🚦 Traffic Rules for {mentioned_city}:**\n\n"""
            for idx, (_, rule) in enumerate(results.iterrows(), 1):
                response += f"""**{idx}. {rule['offense_type']}**\n• **Violation:** {rule['rule_title']}\n• **Fine:** {rule['fine_amount_local']} | **Penalty Points:** {rule['penalty_points']}\n\n"""
            response += f"✅ Found {len(results)} unique violation type(s) in {mentioned_city}."
        else:
            rule = results.iloc[0]
            response = f"""**Based on Indian Traffic Regulations:**\n\n**{rule['rule_title']}**\n\n{rule['paraphrase']}\n\n**Location:** {rule['region']}  \n**Vehicle Type:** {rule['vehicle_type']}\n\n**Fine:** {rule['fine_amount_local']} (≈ ${rule['fine_amount_usd_est']:.2f} USD)  \n**Penalty Points:** {rule['penalty_points']} points\n\n**Enforced By:** {rule['enforcement_agency']}\n\n**Safety Tip:** {rule['preventive_tip']}"""
        
            if len(results) > 1:
                response += f"\n\n✅ Showing {len(results)} unique rules across different cities."
    
    return results, response

def display_rule_card(rule):
    """Display a formatted rule card"""
    confidence = rule.get('similarity', 0.9) * 100
    html_content = f"""
    <div class="rule-card" style="color: white;">
        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; color: white;">
            <div style="flex: 1; min-width: 250px; color: white;">
                <h3 class="rule-title" style="color: white;">{rule['rule_title']}</h3>
                <p style="color: white; font-size: 1.05rem; line-height: 1.7; margin-bottom: 1.5rem;">
                    {rule['paraphrase']}
                </p>
            </div>
            <div style="margin-left: 1rem; margin-top: 0.5rem; color: white;">
                <span class="confidence-badge" style="color: white;">{confidence:.0f}%</span>
                <div style="font-size: 0.7rem; color: white; text-align: center; margin-top: 0.3rem; font-weight: 700;">
                    Relevance
                </div>
            </div>
        </div>
    """

    
    st.markdown(html_content, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box fine-box">
            <div style="font-size: 0.75rem; margin-bottom: 0.5rem; opacity: 0.8;">FINE AMOUNT</div>
            <div style="font-size: 1.5rem; font-weight: 800;">{rule['fine_amount_local']}</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.3rem;">≈ ${rule['fine_amount_usd_est']:.2f} USD</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box points-box">
            <div style="font-size: 0.75rem; margin-bottom: 0.5rem; opacity: 0.8;">PENALTY POINTS</div>
            <div style="font-size: 1.5rem; font-weight: 800;">{rule['penalty_points']}</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.3rem;">Demerit Points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-box region-box">
            <div style="font-size: 0.75rem; margin-bottom: 0.5rem; opacity: 0.8;">REGION</div>
            <div style="font-size: 1.2rem; font-weight: 800;">{rule['region']}</div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.3rem;">{rule['vehicle_type']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="legal-highlight">
            <strong>Legal Reference:</strong> {rule['act_name']}, {rule['section_number']}
        </div>
        
        <div style="margin-top: 1.5rem; display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.9rem; color: {current_theme['text_secondary']};">
            <div><strong style="color: {current_theme['secondary']};">Enforcement:</strong> {rule['enforcement_agency']}</div>
            <div><strong style="color: {current_theme['secondary']};">Applicable:</strong> {rule['applicable_time']}</div>
            <div><strong style="color: {current_theme['secondary']};">Offense:</strong> {rule['offense_type']}</div>
        </div>
        
        <div class="tip-box">
            <div style="font-weight: 700; margin-bottom: 0.8rem; font-size: 1rem;">💡 Safety Tip</div>
            <div style="font-size: 0.9rem; line-height: 1.6;">{rule['preventive_tip']}</div>
        </div>
        
        <div style="margin-top: 1.2rem; font-size: 0.75rem; color: {current_theme['secondary']}; text-align: right; font-weight: 500;">
            Updated: {rule['last_updated']}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        'type': 'bot',
        'content': '''Welcome to the AI Traffic Rules Assistant! 🚦

I use advanced semantic search to provide accurate information about Indian traffic laws, fines, and regulations.

**Smart Features:**
✅ City-specific results when you mention a city
✅ NO duplicate rules - each offense shown only ONCE per city
✅ Accurate vehicle-offense matching

**How to use:**
1. **Mention a city** in your question to get city-specific rules
2. **Skip the city** to see rules from multiple cities
3. Use filters to narrow down results

**Try asking me about traffic violations!**''',
        'timestamp': datetime.now()
    })

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'All'
if 'selected_vehicle' not in st.session_state:
    st.session_state.selected_vehicle = 'All'
if 'selected_offense' not in st.session_state:
    st.session_state.selected_offense = 'All'

# Load data
df = load_data()
if not df.empty:
    vectorizer, tfidf_matrix = initialize_search_engine(df)
else:
    vectorizer, tfidf_matrix = None, None

# Header
st.markdown(f"""
<div class="main-header">
    <h1 class="main-title">🚦 AI Traffic Rules Assistant</h1>
    <p style="font-size:36px; font-weight:700; text-align:center; color:{current_theme['accent']};">
AI That Knows Every Rule Of The Road.
</p>
</div>
""", unsafe_allow_html=True)

# Main Layout
col_main_left, col_main_right = st.columns([2, 1])

with col_main_left:
    # Filter Section
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 **Smart Filters** (Optional)")
    
    col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
    
    with col1:
        if not df.empty:
            st.markdown('<span class="filter-label">City/Region</span>', unsafe_allow_html=True)
            cities = ['All'] + sorted(df['region'].dropna().unique().tolist())
            st.session_state.selected_region = st.selectbox("Region", cities, label_visibility="collapsed", key="region_filter", index=cities.index(st.session_state.selected_region))
    
    with col2:
        if not df.empty:
            st.markdown('<span class="filter-label">Vehicle Type</span>', unsafe_allow_html=True)
            vehicles = ['All'] + sorted(df['vehicle_type'].dropna().unique().tolist())
            selected_vehicle_new = st.selectbox("Vehicle", vehicles, label_visibility="collapsed", key="vehicle_filter", index=vehicles.index(st.session_state.selected_vehicle))
            
            if selected_vehicle_new != st.session_state.selected_vehicle:
                st.session_state.selected_vehicle = selected_vehicle_new
                st.session_state.selected_offense = 'All'
    
    with col3:
        if not df.empty:
            st.markdown('<span class="filter-label">Offense Type</span>', unsafe_allow_html=True)
            all_offenses = ['All'] + sorted(df['offense_type'].dropna().unique().tolist())
            valid_offenses = get_valid_offenses(st.session_state.selected_vehicle, all_offenses)
            
            if st.session_state.selected_offense not in valid_offenses:
                st.session_state.selected_offense = 'All'
            
            st.session_state.selected_offense = st.selectbox("Offense", valid_offenses, 
                                                             index=valid_offenses.index(st.session_state.selected_offense) if st.session_state.selected_offense in valid_offenses else 0,
                                                             label_visibility="collapsed", key="offense_filter")
    
    with col4:
        st.markdown('<span class="filter-label" style="opacity: 0;">.</span>', unsafe_allow_html=True)
        search_filters = st.button("🔍 Search", use_container_width=True, type="primary", key="search_with_filters")
    
    # Show validation warning
    if st.session_state.selected_vehicle != 'All' and st.session_state.selected_offense != 'All':
        two_wheeler = st.session_state.selected_vehicle in ['Motorcycle', 'Bike', 'Two-Wheeler', 'Scooter']
        four_wheeler = st.session_state.selected_vehicle in ['Car', 'Truck', 'Bus', 'Four-Wheeler']
        
        if two_wheeler and ('Seatbelt' in st.session_state.selected_offense):
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Note:</strong> Two-wheelers don't have seatbelt requirements. Please select a different offense type.
            </div>
            """, unsafe_allow_html=True)
        elif four_wheeler and ('Helmet' in st.session_state.selected_offense):
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ Note:</strong> Four-wheelers don't have helmet requirements. Please select a different offense type.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle filter search
    if search_filters:
        valid_search = True
        warning_msg = ""
        
        if st.session_state.selected_vehicle != 'All' and st.session_state.selected_offense != 'All':
            two_wheeler = st.session_state.selected_vehicle in ['Motorcycle', 'Bike', 'Two-Wheeler', 'Scooter']
            four_wheeler = st.session_state.selected_vehicle in ['Car', 'Truck', 'Bus', 'Four-Wheeler']
            
            if two_wheeler and ('Seatbelt' in st.session_state.selected_offense):
                valid_search = False
                warning_msg = "❌ Invalid combination: Two-wheelers don't require seatbelts. Please adjust your filters."
            elif four_wheeler and ('Helmet' in st.session_state.selected_offense):
                valid_search = False
                warning_msg = "❌ Invalid combination: Four-wheelers don't require helmets. Please adjust your filters."
        
        if not valid_search:
            st.session_state.messages.append({
                'type': 'user',
                'content': f"Search: {st.session_state.selected_region} | {st.session_state.selected_vehicle} | {st.session_state.selected_offense}",
                'timestamp': datetime.now()
            })
            st.session_state.messages.append({
                'type': 'bot',
                'content': warning_msg,
                'timestamp': datetime.now()
            })
            st.rerun()
        else:
            filtered_df = df.copy()
            filter_text = []
            
            if st.session_state.selected_region != 'All':
                filtered_df = filtered_df[filtered_df['region'] == st.session_state.selected_region]
                filter_text.append(f"in {st.session_state.selected_region}")
            
            if st.session_state.selected_vehicle != 'All':
                filtered_df = filtered_df[filtered_df['vehicle_type'] == st.session_state.selected_vehicle]
                filter_text.append(f"for {st.session_state.selected_vehicle}")
            
            if st.session_state.selected_offense != 'All':
                filtered_df = filtered_df[filtered_df['offense_type'] == st.session_state.selected_offense]
                filter_text.append(f"related to {st.session_state.selected_offense}")
            
            if not filtered_df.empty:
                query_text = f"Show me traffic rules {' '.join(filter_text)}"
                st.session_state.messages.append({
                    'type': 'user',
                    'content': query_text,
                    'timestamp': datetime.now()
                })
                
                filtered_df['similarity'] = 0.95
                results = deduplicate_results(filtered_df)
                results = results.head(5)
                
                if st.session_state.selected_region != 'All':
                    response = f"✅ Found {len(results)} unique traffic violation type(s) {' '.join(filter_text)}"
                else:
                    response = f"✅ Found {len(results)} unique traffic violation type(s) across cities {' '.join(filter_text)}"
                
                st.session_state.messages.append({
                    'type': 'bot',
                    'content': response,
                    'timestamp': datetime.now()
                })
                
                st.session_state.messages.append({
                    'type': 'rules',
                    'rules': results,
                    'timestamp': datetime.now()
                })
            else:
                st.session_state.messages.append({
                    'type': 'user',
                    'content': f"Show me traffic rules {' '.join(filter_text)}",
                    'timestamp': datetime.now()
                })
                st.session_state.messages.append({
                    'type': 'bot',
                    'content': "❌ No rules found matching your filter criteria. Please try different filters.",
                    'timestamp': datetime.now()
                })
            
            st.rerun()
    
    # Chat Area
    st.markdown("### 💬 **Chat with AI Assistant**")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message['type'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong style="opacity: 0.9;">You:</strong><br>
                    <div style="margin-top: 0.5rem; font-size: 1.05rem;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            elif message['type'] == 'bot':
                st.markdown(f"""
                <div class="bot-message">
                    <strong style="opacity: 0.9;">AI Assistant:</strong><br>
                    <div style="margin-top: 0.5rem; font-size: 1rem; white-space: pre-line;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            elif message['type'] == 'rules':
                for _, rule in message['rules'].iterrows():
                    display_rule_card(rule)
    
    # Suggestions
    st.markdown("### 💡 **Quick Suggestions**")
    
    suggestions = []
    if st.session_state.selected_region != 'All' or st.session_state.selected_vehicle != 'All' or st.session_state.selected_offense != 'All':
        if st.session_state.selected_region != 'All':
            suggestions.append(f"Traffic rules in {st.session_state.selected_region}")
        if st.session_state.selected_vehicle != 'All':
            suggestions.append(f"Violations for {st.session_state.selected_vehicle}")
        if st.session_state.selected_offense != 'All':
            suggestions.append(f"Penalties for {st.session_state.selected_offense}")
    
    if not suggestions:
        suggestions = [
            "Traffic rules in Mumbai",
            "Helmet laws in Delhi",
            "Using mobile while driving",
            "Drunk driving penalties",
            "Red light jumping fine",
            "No seatbelt violation"
        ]
    
    cols = st.columns(min(len(suggestions), 3))
    for idx, suggestion in enumerate(suggestions[:6]):
        with cols[idx % 3]:
            if st.button(suggestion, key=f"sugg_{idx}", use_container_width=True):
                st.session_state.pending_query = suggestion
                st.rerun()
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col_input, col_send = st.columns([5, 1])
    
    with col_input:
        user_input = st.text_input(
            "Type your question here:",
            placeholder="e.g., What are the traffic rules in Mumbai?",
            key="user_input_main",
            label_visibility="collapsed"
        )
    
    with col_send:
        send_button = st.button("🚀 Send", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if send_button and user_input:
        st.session_state.pending_query = user_input
        
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        del st.session_state.pending_query
        
        st.session_state.messages.append({
            'type': 'user',
            'content': query,
            'timestamp': datetime.now()
        })
        
        results, response = process_query(query, df, vectorizer, tfidf_matrix)
        
        st.session_state.messages.append({
            'type': 'bot',
            'content': response,
            'timestamp': datetime.now()
        })
        
        if not results.empty and results['similarity'].iloc[0] >= 0.1:
            st.session_state.messages.append({
                'type': 'rules',
                'rules': results,
                'timestamp': datetime.now()
            })
        
        st.rerun()

with col_main_right:
    st.markdown("### 📊 **Database Statistics**")
    
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">Total Rules</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cities_count = df['region'].nunique()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{cities_count}</div>
                <div class="stat-label">Cities Covered</div>
            </div>
            """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            vehicle_count = df['vehicle_type'].nunique()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{vehicle_count}</div>
                <div class="stat-label">Vehicle Types</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            offense_count = df['offense_type'].nunique()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{offense_count}</div>
                <div class="stat-label">Offense Types</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 **About**")
    
    st.markdown(f"""
    <div style="background: {current_theme['card_bg']}; padding: 1.5rem; border-radius: 15px; border: 2px solid {current_theme['card_border']}; box-shadow: 0 4px 15px {current_theme['shadow_color']};">
        <p style="color: {current_theme['text_secondary']}; font-weight: 600; margin-bottom: 1rem;">
            <strong>AI-Powered Search Engine</strong>
        </p>
        <p style="color: {current_theme['text_primary']}; font-size: 0.9rem; line-height: 1.6;">
            This chatbot uses <strong>TF-IDF vectorization</strong> and <strong>cosine similarity</strong> 
            to provide accurate traffic rule information across India.
        </p>
        <hr style="border: none; border-top: 1px solid {current_theme['card_border']}; margin: 1rem 0;">
        <p style="color: {current_theme['text_secondary']}; font-weight: 600; margin-bottom: 0.5rem;">
            <strong>✨ Key Features:</strong>
        </p>
        <ul style="color: {current_theme['text_primary']}; font-size: 0.85rem; line-height: 1.8; padding-left: 1.2rem;">
            <li><strong>City-aware search</strong> - Mention a city to get specific results</li>
            <li><strong>No duplicates</strong> - Each offense shown only ONCE per city</li>
            <li><strong>Vehicle validation</strong> - Prevents invalid combinations</li>
            <li><strong>Semantic search</strong> - Understands natural language</li>
            <li><strong>Day/Night mode</strong> - Toggle theme for comfort</li>
            <li><strong>Comprehensive data</strong> - Fines, penalties, and tips</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚠️ **Disclaimer**")
    
    st.markdown("""
    <div style="background: #fff9e6; padding: 1rem; border-radius: 12px; border: 2px solid #f59e0b; font-size: 0.8rem; color: #92400e; line-height: 1.6;">
        <strong>Legal Notice:</strong><br>
        Information provided is for reference only. 
        Traffic rules and fines may vary by location and are subject to change. 
        Always verify with official authorities.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚡ **Quick Actions**")
    
    if st.button("🔄 Clear Chat History", use_container_width=True, key="clear_chat"):
        st.session_state.messages = [{
            'type': 'bot',
            'content': 'Chat cleared! How can I assist you with traffic rules today?',
            'timestamp': datetime.now()
        }]
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {current_theme['secondary']}; font-size: 0.75rem; margin-top: 1rem;">
        <p><strong>🚦 AI Traffic Rules Assistant</strong></p>
        <p>Powered by Machine Learning & NLP</p>
        <p>© 2025 - Stay Safe on Indian Roads</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="color: {current_theme['text_secondary']};">📌 Navigation</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 **How to Use**")
    st.markdown(f"""
    <div style="background: {current_theme['card_bg']}; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border: 2px solid {current_theme['card_border']};">
        <ol style="font-size: 0.9rem; line-height: 2; color: {current_theme['text_primary']};">
            <li><strong>Mention a city</strong> - "Traffic rules in Mumbai"</li>
            <li><strong>Ask naturally</strong> - The AI understands you</li>
            <li><strong>No duplicates</strong> - Each offense shown only once</li>
            <li><strong>Use filters</strong> - For more specific results</li>
            <li><strong>Toggle theme</strong> - Click {theme_icon} at top right</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔥 **Popular Queries**")
    popular = [     
        "Traffic rules in Mumbai",
        "Helmet rule Delhi",
        "Drunk driving penalty",
        "Mobile phone usage",
        "Red light violation",
        "No parking fine"
    ]
    
    for idx, query in enumerate(popular):
        if st.button(f"💬 {query}", key=f"popular_{idx}", use_container_width=True):
            st.session_state.pending_query = query
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 📞 **Emergency Contacts**")
    st.markdown(f"""
   <div style="background: {current_theme['card_bg']}; padding: 1rem; border-radius: 10px; border: 2px solid {current_theme['card_border']}; color: white;">
    <p style="margin: 0.5rem 0; color: #64113f; font-weight: 600;"><strong>🚨 Traffic Police:</strong> 100</p>
    <p style="margin: 0.5rem 0; color: #64113f; font-weight: 600;"><strong>🚑 Ambulance:</strong> 108</p>
    <p style="margin: 0.5rem 0; color: #64113f; font-weight: 600;"><strong>🔥 Fire:</strong> 101</p>
    <p style="margin: 0.5rem 0; color: #64113f; font-weight: 600;"><strong>👮 Women Helpline:</strong> 1091</p>
</div>

    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 **Pro Tips**")
    st.info(f"""
    **Smart Search:** Mention a city name to get results for that specific city only!
    
    **Theme Toggle:** Click the {theme_icon} button at the top right to switch between day and night modes!
    """)