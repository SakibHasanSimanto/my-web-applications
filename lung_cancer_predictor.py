import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page configuration
st.set_page_config(
    page_title="ফুসফুস ক্যান্সার নির্ণায়ক ১.০",
    page_icon=":lungs:",  # Optional icon
    layout="wide",  # Use wide layout
    initial_sidebar_state="expanded"  # Sidebar expanded by default
)

# Initialize session state for outcome
if "outcome" not in st.session_state:
    st.session_state.outcome = None

st.title('ফুসফুস ক্যান্সার নির্ণায়ক ১.০')
st.write('অনুগ্রহ করে নিম্নোক্ত নির্দেশাবলী অনুসরণ করে আপনার ডেটা ইনপুট দিন')

# Instructions
if st.button('নির্দেশাবলী'):
    arr1 = ['লিঙ্গ', 'বয়স', 'ধূমপান', 'হলুদাভ ত্বক', 'দুশ্চিন্তা', 'সামাজিক চাপ',
            'দীর্ঘস্থায়ী রোগ', 'ক্লান্তি', 'এলার্জি', 'হুইজিং', 'মদ্যপান', 'কাশি',
            'শ্বাসকষ্ট', 'গলাধঃকরণ জটিলতা', 'বুক ব্যথা']
    arr2 = ['আপনার লিঙ্গ?', 'বয়স?', 'ধূমপান করেন?', 'ত্বক/আংগুল মাঝেমধ্যে হলুদ হয়ে যায়?',
            'দুশ্চিন্তা করেন?', 'সহকর্মীদের থেকে নেতিবাচক কোনো চাপ অনুভব করেন?',
            'দীর্ঘমেয়াদী কোনো অসুখ আছে?', 'ক্লান্তি অনুভব করেন?', 'এলার্জি আছে?',
            'শ্বাস নেবার সময় বুকের ভেতরে শব্দ হয়?', 'এলকোহলজাতীয় কিছু খান?', 'কাশি আছে?',
            'শ্বাসকষ্ট আছে?', 'খাবার গিলতে কষ্ট হয়?', 'বুকে ব্যথা আছে?']
    arr3 = ['মহিলা (0) / পুরুষ(1)', 'প্র/ন', 'করেন(2) / করেন না(1)', 'হয়(2) / হয়না(1)',
            'করেন(2) / করেন না(1)', 'করেন(2) / করেন না(1)', 'আছে(2) / নেই(1)',
            'করেন(2) / করেন না(1)', 'আছে(2) / নেই(1)', 'হয়(2) / হয়না(1)',
            'খান(2) / খান না(1)', 'আছে(2) / নেই(1)', 'আছে(2) / নেই(1)',
            'হয়(2) / হয়না(1)', 'আছে(2) / নেই(1)']

    info_df = pd.DataFrame({'প্রভাবক': arr1, 'প্রশ্ন': arr2, 'অপশন': arr3})
    st.write(info_df)

# Input fields
st.write('আপনার ডেটা ইনপুট দিন')
col1, col2 = st.columns(2)

with col1:
    gender = st.text_input('লিঙ্গ:')
    age = st.text_input('বয়স:')
    smoking = st.text_input('ধূমপান করেন?')
    ylf = st.text_input('ত্বক/আংগুল মাঝেমধ্যে হলুদ হয়ে যায়?')
    anx = st.text_input('দুশ্চিন্তা করেন?')
    peer_press = st.text_input('সহকর্মীদের থেকে নেতিবাচক কোনো চাপ অনুভব করেন?')

with col2:
    chron_dis = st.text_input('দীর্ঘমেয়াদী কোনো অসুখ আছে?')
    fatig = st.text_input('ক্লান্তি অনুভব করেন?')
    allerg = st.text_input('এলার্জি আছে?')
    wheez = st.text_input('শ্বাস নেবার সময় বুকের ভেতরে শব্দ হয়?')
    alchl = st.text_input('এলকোহলজাতীয় কিছু খান?')
    cough = st.text_input('কাশি আছে?')
    breath = st.text_input('শ্বাসকষ্ট আছে?')
    sd = st.text_input('খাবার গিলতে কষ্ট হয়?')
    chest = st.text_input('বুকে ব্যথা আছে?')

# Predict button
if st.button('Predict'):
    try:
        input_list = [gender, age, smoking, ylf, anx, peer_press,
                      chron_dis, fatig, allerg, wheez, alchl,
                      cough, breath, sd, chest]
        prediction = model.predict([input_list])
        if prediction[0] == 1:
            st.session_state.outcome = 1
            st.success('ফুসফুসে ক্যান্সারের সম্ভাবনাঃ আছে')
        else:
            st.session_state.outcome = 0
            st.success('ফুসফুসে ক্যান্সারের সম্ভাবনাঃ নেই')
            st.balloons()

    except Exception as e:
        st.error(f"Error: {e}")

# Comments button
if st.button('মন্তব্য'):
    try:
        if st.session_state.outcome == 1:
            st.write("### পূর্বাভাস: ফুসফুসের ক্যান্সার শনাক্ত হতে পারে")
            st.write("মডেলের উপর ভিত্তি করে স্বাস্থ্য পরামর্শ নিন।")
        elif st.session_state.outcome == 0:
            st.write("### পূর্বাভাস: ফুসফুসের ক্যান্সার শনাক্ত হয়নি")
            st.write("সুস্থ থাকার জন্য জীবনধারা সচেতন থাকুন।")
        else:
            st.warning("অনুগ্রহ করে প্রথমে 'Predict' ক্লিক করুন।")

    except Exception as e:
        st.error(f"Error: {e}")
