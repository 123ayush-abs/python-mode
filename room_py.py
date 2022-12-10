import pandas as pd  # for dataset cleaning and analysis
import streamlit as st
from PIL import Image
import time
import numpy as np  # for handling large array
import difflib  # find the closest match between user choice and dataset values
# convert textual values to numerical for comparing
from sklearn.feature_extraction.text import TfidfVectorizer
# finding the similarity b/w user choicxe and dataset values
from sklearn.metrics.pairwise import cosine_similarity
img = Image.open("hostel.png")
st.set_page_config(page_title="The Room Plaza", page_icon=img)
st.image(img, width=70)
st.title("THE ROOM PLAZA")
st.markdown("Relocation made easy!!")
hostel = pd.read_csv('Hostel.csv')  # read datset
hostel.drop_duplicates()  # drop duplicates
# print(hostel.head())#dataset set preview
selected_features = ['city', 'area',
                     'animal_allowance', 'rent_amount', 'furniture']
for feature in selected_features:
    hostel[feature] = hostel[feature].fillna('')  # handling null values
combined = hostel['city']+' '+hostel['area']+' '+hostel['animal_allowance']+' ' + \
    str(hostel['rent_amount']) + ' ' + \
    hostel['furniture']  # combibing all the parameters for predictions
# print(combined)
vectorizer = TfidfVectorizer()
# convering the features into numerical values
feature_vectors = vectorizer.fit_transform(combined)
# print(feature_vectors)
# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)


def predcit_hostel(city_name, animalall, rent_range, total_rooms):
    animal_allow = ""
    if (animalall == 'Yes'):
        animal_allow = "acept"
    else:
        animal_allow = "not acept"

    # getting list of all city in dataset
    list_of_all_city = hostel['city'].tolist()
    find_close_match = difflib.get_close_matches(city_name, list_of_all_city)
    close_match = find_close_match[0]
    index_of_the_city = hostel[hostel.city == close_match]['index'].values
    similarity_score = list(enumerate(similarity[index_of_the_city]))
    sorted_similar_city = sorted(
        similarity_score, key=lambda x: x[1].any(), reverse=True)
    i = 1
    list_area = []
    list_rent = []
    list_room = []
    for city in sorted_similar_city:
        index = city[0]
        title_from_index = hostel[hostel.index == index]['area'].values[0]
        city_from_index = hostel[hostel.index == index]['city'].values[0]
        rent_from_index = hostel[hostel.index ==
                                 index]['rent_amount'].values[0]
    # list_rent.append(hostel[hostel.index==index]['rent_amount'].values[0])
        animal_from_index = hostel[hostel.index ==
                                   index]['animal_allowance'].values[0]
        room_from_index = hostel[hostel.index == index]['rooms'].values[0]
        # list_room.append(hostel[hostel.index==index]['rooms'].values[0])
        if (i < 100 and animal_from_index == animal_allow and rent_from_index <= rent_range and city_from_index == city_name and room_from_index == total_rooms):
            list_area.append(title_from_index)
            list_rent.append(rent_from_index)
            list_room.append(room_from_index)
            # st.write(title_from_index)
            i += 1
    return list_area, list_rent


status = st.radio(
    'Select City:', ['Delhi', 'Banglore', 'Chennai', 'Kolkata', 'Mumbai'])
animalallow = st.selectbox("Animal Allow: ",
                           ['Yes', 'No'])
roomsnum = st.slider("Select number of rooms:", 1, 3)
rentrange = st.selectbox("Rent Amount: ",
                         [5000, 10000, 12000, 15000])

if (st.button("Predict Room!!")):
    my_bar = st.progress(0)
    for percent_complete in range(100):
        # time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    with st.sidebar:
        st.title("PGs Suggested For You:--")
        lp, lp1 = predcit_hostel(str(status), str(
            animalallow), int(rentrange), int(roomsnum))
        for i in range(len(lp)):
            st.info("Located In="+str(lp[i]))
            st.warning("Total Rent to be paid="+str(lp1[i]))
