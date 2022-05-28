import json
import cv2
import cv2 as cv
import math
import time
import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pandas.core.reshape.pivot import pivot_table
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import pymongo
import hashlib

# defining user_name
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''

# logo and app icon
logo = Image.open('logo.png')
st.set_page_config(page_title="bookSpot", page_icon=logo, layout="wide")

# hiding non required menu icon
hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hiddes; }
            </style>
            """
st.markdown(hide_menu_style, unsafe_allow_html=True)

client = pymongo.MongoClient(
    'mongodb+srv://user1:3N7AkEaDqXCgKr2w@cluster0.fdq4e.mongodb.net/?retryWrites=true&w=majority'
)

db = client.get_database('main').user1
search_db = client.get_database('searches').data
dbCart = client.get_database('cart').books
ratingDB = client.get_database('cart').rate

# importing required datas
# dataset for hybrid based recommendation
hybrid_content_dict = pickle.load(open('hybrid_content.pkl', 'rb'))
hybrid_content_df = pd.DataFrame(hybrid_content_dict)

hybrid_collab_dict = pickle.load(open('hybrid_collab.pkl', 'rb'))
hybrid_collab_df = pd.DataFrame(hybrid_collab_dict)

# dataset for content based recommendation
content_dict = pickle.load(open('content_book.pkl', 'rb'))
content_df = pd.DataFrame(content_dict)
content_df.sort_values(by='title', inplace=True, kind='quicksort')
content_df.reset_index(inplace=True)
content_df.drop(['index'], axis=1, inplace=True)

# dataset for item to item collaborative filter based recommendation
collab_dict = pickle.load(open('book_dict.pkl', 'rb'))
collab_df = pd.DataFrame(collab_dict)

# datasets for specifically filter based recommendation
nonfiction_dict = pickle.load(open('nonfiction_dict.pkl', 'rb'))
nonfiction_df = pd.DataFrame(nonfiction_dict)

child_dict = pickle.load(open('child_dict.pkl', 'rb'))
child_df = pd.DataFrame(child_dict)

science_fiction_dict = pickle.load(open('sciencefiction_dict.pkl', 'rb'))
science_fiction_df = pd.DataFrame(science_fiction_dict)

fantasy_dict = pickle.load(open('fantasy_dict.pkl', 'rb'))
fantasy_df = pd.DataFrame(fantasy_dict)

axn_adv_dict = pickle.load(open('axn_adv_dict.pkl', 'rb'))
axn_adv_df = pd.DataFrame(axn_adv_dict)

politic_dict = pickle.load(open('politic_dict.pkl', 'rb'))
politic_df = pd.DataFrame(politic_dict)

crime_thrill_dict = pickle.load(open('crime_thrill_dict.pkl', 'rb'))
crime_thrill_df = pd.DataFrame(crime_thrill_dict)

# dataset for age-gender based recommendation
age_based_dict = pickle.load(open('age_based_filter.pkl', 'rb'))
age_based_df = pd.DataFrame(age_based_dict)
grouped = age_based_df.groupby(age_based_df.category)

senior_women_df = grouped.get_group("senior women")
senior_men_df = grouped.get_group("senior men")
adult_guys_df = grouped.get_group("Young adults guys")
women_df = grouped.get_group("women")
men_df = grouped.get_group("men")
teenage_girls_df = grouped.get_group("teenage girls")
teenage_boys_df = grouped.get_group("teenage boys")
girl_child_df = grouped.get_group("girl child")
boy_child_df = grouped.get_group("boy child")
adult_girls_df = grouped.get_group("Young adult girls")

# navigation bar
tab = option_menu(
    menu_title=None,
    options=["Login", "Home", "Guess Me", "Book Shelf", "Search", "Books for You", "Go to Cart", "Search History",
             "About us", "LogOut"],
    icons=["box-arrow-in-right", "house", "emoji-laughing", "bookshelf", "search", "book", "cart3", "clock-history",
           "people-fill",
           "box-arrow-right"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "Crayola Red"},
        "icon": {"color": "Folly", "font-size": "25px"},
    }
)

# login page
if tab == "Login":
    def make_hashes(password):
        return hashlib.sha256(str.encode(password)).hexdigest()


    def check_hashes(password, hashed_text):
        if make_hashes(password) == hashed_text:
            return hashed_text
        return False


    if "login" not in st.session_state:
        st.session_state["login"] = False

    # adding user data to our record
    def add_userdata(username, password):
        record = {
            'username': username,
            'password': password
        }
        db.insert_one(record)


    def valid_user(username, password):
        record = {
            'username': username,
            'password': password
        }
        if (db.count_documents(record, limit=1)) >= 1:
            return True
        else:
            return False

    #if user name is not taken by other person we'll give warning
    def user_name_available(username):
        if (db.count_documents({'username': username}, limit=1)) < 1:
            return True
        else:
            return False


    def logged_in():
        st.title("You're good to go!")

        def load_lottieurl(url: str):
            r = requests.get(url)
            if (r.status_code != 200):
                return None
            return r.json()

        lottie_login = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_hjx5yvar.json")
        st_lottie(
            lottie_login,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=600,
            width=None,
            key="login",
        )


    def main():
        if st.session_state["user_name"]:
            st.session_state["login"] = True
        menu = ["Login", "SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)

        if st.session_state['login']:
            logged_in()
        elif choice == "Login":
            st.subheader("Login Section")

            username = st.text_input("User Name")
            password = st.text_input("Password", type='password')
            if st.button("Login"):
                result = valid_user(username, make_hashes(password))
                if result:
                    st.session_state['login'] = True
                    st.session_state['user_name'] = username
                    st.success("Logged In as {}".format(username))

                    logged_in()

                else:
                    st.warning("Incorrect Username/Password")


        elif choice == "SignUp":
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            unique_username = user_name_available(new_user)
            if not unique_username:
                st.warning("Username taken, please select a new Username")
            new_password = st.text_input("Password", type='password')

            if st.button("Signup"):
                if (new_user == '') or (new_password == ''):
                    st.warning("Some fields missing")

                else:
                    add_userdata(new_user, make_hashes(new_password))
                    st.success("You have successfully created a valid Account")
                    st.info("Go to Login Menu to login")


    if __name__ == '__main__':
        main()

# Home page
if tab == "Home":
    # defining function for animation
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()


    colm = st.columns(17)
    colm[7].image('logo.png', width=100)
    colm[8].title("bookSpot")
    lottie_welcome = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_1TcivY.json")
    st_lottie(
        lottie_welcome,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height=580,
        width=None,
        key="Home",
    )

# Age and Gender detection page
if tab == "Guess Me":
    # it will recommend books based on age and gender
    def recommend_age_gender_based(user_age, gender):
        final_df = pd.DataFrame()
        if user_age <= 14:
            if gender == 'Female':
                final_df = girl_child_df
            else:
                final_df = boy_child_df
        elif user_age <= 18:
            st.write(gender)
            if gender == 'Female':
                final_df = teenage_girls_df
            else:
                final_df = teenage_boys_df
        elif user_age <= 24:
            if gender == 'Female':
                final_df = adult_girls_df
            else:
                final_df = adult_guys_df
        elif user_age <= 60:
            if gender == 'Female':
                final_df = women_df
            else:
                final_df = men_df
        else:
            if gender == 'Female':
                final_df = senior_women_df
            else:
                final_df = senior_men_df
        cols = st.columns(5)
        for i in range(5):
            cols[i].text(final_df.iloc[i][0])
            cols[i].image(final_df.iloc[i][5], caption=None, width=200)

        for i in range(5):
            with cols[i]:
                expander = st.expander("Overview")
                expander.write(final_df.iloc[i][2])


    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        return frameOpencvDnn, bboxes


    faceProto = "modelNweight/opencv_face_detector.pbtxt"
    faceModel = "modelNweight/opencv_face_detector_uint8.pb"

    ageProto = "modelNweight/age_deploy.prototxt"
    ageModel = "modelNweight/age_net.caffemodel"

    genderProto = "modelNweight/gender_deploy.prototxt"
    genderModel = "modelNweight/gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load network
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    padding = 20


    def age_gender_detector(frame):
        # Read frame
        t = time.time()
        age = 0
        gender = "None"
        frameFace, bboxes = getFaceBox(faceNet, frame)
        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                   max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()].split('-')
            age = ((int(age[0][1:]) + int(age[1][:-1])) / 2)
        dict = {
            "age": age,
            "gender": gender
        }
        return dict


    if "gender2" not in st.session_state:
        st.session_state.gender2 = "Female"
    if "age2" not in st.session_state:
        st.session_state.age2 = 0
    col1, col2 = st.columns(2)
    with col1:
        st.title("Let's have some fun😄")
        st.write(
            """##### Please look into the camera with proper light on the face so that we can detect your age and gender with better accuracy.""")

        FRAME_WINDOW = st.image([])
        cam = cv2.VideoCapture(0)
        my_table = st.table()
        age = 0
        gender = "None"
        age1 = 0
        gender1 = "M"

        #this will detect the users age and gender
        def detect_age_gender(flag):
            global age
            global gender
            predicted_age = 0
            predicted_gender = "M"
            timeout = time.time() + 5
            while flag & (time.time() <= timeout):
                ret, frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)

                new_dict = age_gender_detector(frame)
                predicted_gender = new_dict["gender"]
                predicted_age = new_dict["age"]
                data = {'Gender': [predicted_gender],
                        'Age': [predicted_age]}
                df = pd.DataFrame(data)
                my_table.dataframe(df)
            dict = {"age": predicted_age, "gender": predicted_gender}
            return dict


        flag = False
        if st.button("start"):
            flag = True

            dict = detect_age_gender(flag)
            data = {'Gender': [dict["gender"]],
                    'Age': [dict["age"]]}
            df = pd.DataFrame(data)
            my_table.dataframe(df)
            cam.release()
            cv2.destroyAllWindows()
            st.session_state.age2 = dict["age"]
            st.session_state.gender2 = dict["gender"]
    result_btn = st.button('result')
    if "result_state" not in st.session_state:
        st.session_state.result_state = False
    if result_btn or st.session_state.result_state:
        st.session_state.result_state = True
        option = ["Yup!😁", "No"]
        slct = st.selectbox("Did we guess it right?", option)
        if "select_state" not in st.session_state:
            st.session_state.select_state = False

        if slct or st.session_state.select_state:
            st.session_state.select_state = True
        if "yes_state" not in st.session_state:
            st.session_state.yes_state = False
        if "no_state" not in st.session_state:
            st.session_state.no_state = False
        if slct == "Yup!😁" or st.session_state.yes_state:
            recommend_age_gender_based(st.session_state.age2, st.session_state.gender2)
            st.write("yahoo🥳")
            st.write(st.session_state.age2, st.session_state.gender2)
        elif slct == "No" or st.session_state.no_state:
            gender_option = ["Male", "Female"]
            st.session_state.gender2 = st.selectbox("Please enter your gender:", gender_option)
            st.session_state.age2 = st.number_input("Please enter your age:", min_value=1, value=24)
            recommend_age_gender_based(st.session_state.age2, st.session_state.gender2)

    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if (r.status_code != 200):
                return None
            return r.json()


        lottie_guess = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_l6jf9iln.json")
        st_lottie(
            lottie_guess,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=700,
            width=800,
            key="guessme",
        )

# Books for you page
if tab == "Books for You":
    # creating a pivot table so that we can feed it to knn model
    hybrid_df_pivot = hybrid_collab_df.pivot_table(columns='user_id', index='title', values='rating')

    # replacing NaN values with 0
    hybrid_df_pivot.fillna(0, inplace=True)

    # creating a sparse matrix because we have lots of 0s in our pivot table, that can increase time complexity
    # unnecessarily
    hybrid_df_sparse = csr_matrix(hybrid_df_pivot)

    # creating the model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)

    # training the model
    knn_model.fit(hybrid_df_pivot)

    # function to estimate rating
    def cal_rating(sim_score, rating):
        global numerator
        global denominator
        global x
        numerator = 0
        denominator = 0
        if len(rating) == 0:
            x = sim_score
        else:
            for i in range(len(sim_score)):
                numerator += sim_score[i] * rating[i]
                denominator += sim_score[i]
            if denominator == 0:
                x = 3
            else:
                x = (numerator / denominator)
        return x

    # this function will recommend books based on what's in your cart and browse history, it will first estimate the ratings
    def hybrid_recommend_history_based(cart, rated, browse_hist):
        if not cart or not browse_hist:
            st.write(" ##### We don't have enough history of your activity to recommend you books.")

            def load_lottieurl(url: str):
                r = requests.get(url)
                if r.status_code != 200:
                    return None
                return r.json()

            lottie_no_history = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_cwzd46cu.json")
            st_lottie(
                lottie_no_history,
                speed=1,
                reverse=False,
                loop=True,
                quality="low",
                height=600,
                width=None,
                key="history",
            )

        else:
            books_dict = {}
            books_sim_cart = []
            for item in cart:
                book_idx = np.where(hybrid_df_pivot.index == item)[0][0]
                distances, indices = knn_model.kneighbors(hybrid_df_pivot.iloc[book_idx, :].values.reshape(1, -1),
                                                          n_neighbors=6)
                for i in range(indices.shape[1]):
                    if (i != 0):
                        item = hybrid_df_pivot.index[indices[0][i]]
                        if not (item in books_dict) and not (item in cart):
                            books_dict[item] = 0.6
                            books_sim_cart.append(hybrid_df_pivot.index[indices[0][i]])
            books_sim_browse = []
            for item in browse_hist:
                book_idx = np.where(hybrid_df_pivot.index == item)[0][0]
                distances, indices = knn_model.kneighbors(hybrid_df_pivot.iloc[book_idx, :].values.reshape(1, -1),
                                                          n_neighbors=11)
                for i in range(indices.shape[1]):
                    if i != 0:
                        item = hybrid_df_pivot.index[indices[0][i]]
                        if not (item in books_dict) and not (item in cart):
                            books_dict[item] = 0.4
                            books_sim_browse.append(hybrid_df_pivot.index[indices[0][i]])

            if len(books_sim_browse) == 0:
                st.write("We don't have your enough browse history to recommend you books.")

            temp_df = pd.DataFrame.from_dict(books_dict, orient='index')
            temp_df.rename(columns={temp_df.columns[0]: 'weightage'}, inplace=True)
            temp_df.reset_index(inplace=True)
            temp_df.rename(columns={'index': 'title'}, inplace=True)

            index_rated = []
            for item in rated:
                temp_df.loc[len(temp_df.index)] = [item, 0]
                index_rated.append(len(temp_df.index) - 1)

            temp_df = temp_df.merge(hybrid_content_df, how='left', on='title')

            vectorizer = CountVectorizer(max_features=2500, stop_words='english')
            vector_form = vectorizer.fit_transform(temp_df['tags']).toarray()

            similarity = cosine_similarity(vector_form)

            rated_books = []
            for book in rated:
                rated_books.append(book)

            sim_cart = pd.DataFrame(columns=rated_books, index=books_sim_cart)
            sim_browsed = pd.DataFrame(columns=rated_books, index=books_sim_browse)

            for i in range(sim_cart.shape[0]):
                for j in range(sim_cart.shape[1]):
                    row_idx = temp_df[temp_df['title'] == sim_cart.index[i]].index[0]
                    col_idx = temp_df[temp_df['title'] == sim_cart.columns[j]].index[0]
                    sim_cart.iloc[i][j] = similarity[row_idx][col_idx]

            for i in range(sim_browsed.shape[0]):
                for j in range(sim_browsed.shape[1]):
                    row_idx = temp_df[temp_df['title'] == sim_browsed.index[i]].index[0]
                    col_idx = temp_df[temp_df['title'] == sim_browsed.columns[j]].index[0]
                    sim_browsed.iloc[i][j] = similarity[row_idx][col_idx]

            sim_cart['est_rating'] = " "
            sim_browsed['est_rating'] = " "

            sim_cart.reset_index(inplace=True)
            sim_browsed.reset_index(inplace=True)

            sim_cart.rename(columns={'index': 'target_books'}, inplace=True)
            sim_browsed.rename(columns={'index': 'target_books'}, inplace=True)

            for i in range(sim_cart.shape[0]):
                cart_sim_score = []
                rating = []
                for j in range(1, sim_cart.shape[1] - 1):
                    cart_sim_score.append(sim_cart.iat[i, j])
                    rating.append(rated[sim_cart.columns[j]])
                x = cal_rating(cart_sim_score, rating)
                sim_cart.at[i, 'est_rating'] = x

            for i in range(sim_browsed.shape[0]):
                browse_sim_score = []
                rating = []
                for j in range(1, sim_browsed.shape[1] - 1):
                    browse_sim_score.append(sim_browsed.iat[i, j])
                    rating.append(rated[sim_browsed.columns[j]])
                x = cal_rating(browse_sim_score, rating)
                sim_browsed.at[i, 'est_rating'] = x

            sim_cart.sort_values(by=['est_rating'], ascending=False, inplace=True, kind='quicksort')
            sim_browsed.sort_values(by=['est_rating'], ascending=False, inplace=True, kind='quicksort')

            final_cart_df = sim_browsed.filter(['target_books', 'est_rating'])
            final_cart_df.reset_index(inplace=True)
            final_cart_df.drop(['index'], axis=1, inplace=True)
            final_cart_df.rename(columns={'target_books': 'title'}, inplace=True)

            final_browse_df = sim_cart.filter(['target_books', 'est_rating'])
            final_browse_df.reset_index(inplace=True)
            final_browse_df.drop(['index'], axis=1, inplace=True)
            final_browse_df.rename(columns={'target_books': 'title'}, inplace=True)

            final_cart_df = final_cart_df.merge(hybrid_content_df, how='left', on='title')
            final_browse_df = final_browse_df.merge(hybrid_content_df, how='left', on='title')

            final_cart_df.drop(['tags'], axis=1, inplace=True)
            final_browse_df.drop(['tags'], axis=1, inplace=True)

            st.title("Top picks for you!")
            st.write(""" ##### Based on what's in your cart: """)
            hybrid_col = st.columns(5)
            for y in range(5):
                hybrid_col[y].image(final_cart_df.iloc[y][6], caption=final_cart_df.iloc[y][0], width=200)

            st.write(""" ##### Based on what's in your search history: """)
            hybrid_col = st.columns(5)
            for y in range(5):
                hybrid_col[y].image(final_browse_df.iloc[y][6], caption=final_browse_df.iloc[y][0], width=200)


    filter_by = {'username': st.session_state['user_name']}
    num = search_db.count_documents(filter_by)
    cart = []
    browse_hist = []
    rated = {}
    filter_by = {'username': st.session_state['user_name']}
    cart_count = dbCart.count_documents(filter_by)
    browse_count = dbCart.count_documents(filter_by)
    c = cart_count
    b = browse_count
    obj_cart = dbCart.find({'username': st.session_state['user_name']})
    obj_browse = search_db.find({'username': st.session_state['user_name']})

    if b <= 5:
        for obj in search_db.find({'username': st.session_state['user_name']}):
            browse_hist.append(obj["book"])
    else:
        for k in range(b - 1, b - 5 - 2, -1):
            browse_hist.append(obj_browse[k]["book"])
    if c <= 5:
        for obj in dbCart.find({'username': st.session_state['user_name']}):
            cart.append(obj["book"])
    else:
        for k in range(c - 1, c - 5 - 2, -1):
            cart.append(obj_cart[k]["book"])

    for obj in ratingDB.find({'username': st.session_state['user_name']}):
        rated[obj["book"]] = obj["rating"] - 5
    # we are scaling down the rating from range (1-10) to (-5 to 5), if the rating is below 5 in (1-10) scale we will
    # consider that the user dislikes the book

    hybrid_recommend_history_based(cart, rated, browse_hist)

# About us page
if tab == "About us":
    st.title("About us")


    def load_lottieurl(url: str):
        r = requests.get(url)
        if (r.status_code != 200):
            return None
        return r.json()


    lottie_about = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_arirrjzh.json")
    lottie_coding = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_o6spyjnc.json")
    cols1, cols2, cols3 = st.columns(3)
    with cols1:
        st_lottie(lottie_coding, loop=True, quality="low", height=550)
    with cols2:
        st.write(""" ##### Bonjour!! """)
        st.write(
            """ ##### I am Bharti Patel, a 2nd year BTech student at IIT (ISM) Dhanbad.""")
        st.write(
            """ ##### I've created this app with Python as my langugage, Machine Learning for model, Pycharm as IDE, Streamlit as framework, MongoDB as database and Heroku for Deployment.""")
        st.write(
            """ ##### I've used the K-Nearest Neighbor algorithm for collaborative-based filtering and the Vector Space Model for content-based filtering. I have also used hybrid filtering to track users' behavior and recommend them accordingly. For sorting, I have used different algorithms, like the quicksort and Timsort algorithms. Similarly, I have used algorithms like membership operators and linear search for searching. For scoring and ranking, I have estimated the scores of different items and ranked them.""")
        st.write(
            """ ##### This web app is built under the mentorship program offered by Microsoft, Microsoft Engage 2022. I was mentored by Ekta Mehla ma'am throughout this program .""")
    with cols3:
        st_lottie(lottie_about, loop=True, quality="low", height=550)

# Recommendation based on user history

# Cart page
if tab == "Go to Cart":
    def load_lottieurl(url: str):
        r = requests.get(url)
        if (r.status_code != 200):
            return None
        return r.json()


    filter_by = {'username': st.session_state['user_name']}
    items_cart_num = dbCart.count_documents(filter_by)
    if items_cart_num == 0:
        lottie_cart = load_lottieurl("https://assets1.lottiefiles.com/temp/lf20_jzqS18.json")
        st.title("Your Cart is Empty!")
        st_lottie(
            lottie_cart,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=650,
            width=None,
            key="empty_cart"
        )
    else:
        rows = 1
        if items_cart_num > 5:
            if items_cart_num % 5 == 0:
                rows = int(items_cart_num / 5)
            else:
                rows = int(items_cart_num / 5) + 1

        x = dbCart.find({'username': st.session_state['user_name']})
        k = items_cart_num - 1
        for i in range(rows):
            col = st.columns(5)
            for j in range(5):
                if k >= 0:
                    # searching
                    # algo : membership operators for boolean return(O(n)), linear search for .index[] method O(1)
                    idx = content_df[content_df['title'] == x[k]["book"]].index[0]
                    col[j].image(content_df.iloc[idx][6], caption=content_df.iloc[idx][1], width=200)
                    if col[j].button("Remove", key=content_df.iloc[idx][1]):
                        dbCart.delete_one({
                            'username': st.session_state['user_name'],
                            'book': x[k]["book"]
                        })
                        num = items_cart_num - 1;
                    else:
                        label = False
                    k -= 1

        lottie_cart = load_lottieurl("https://assets2.lottiefiles.com/private_files/lf30_x2lzmtdl.json")
        st_lottie(
            lottie_cart,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=650,
            width=None,
        )
# LogOut page
if tab == "LogOut":
    st.title("Logout from bookSpot?")
    col1, col2 = st.columns(2)
    with col2:
        def load_lottieurl(url: str):
            r = requests.get(url)
            if (r.status_code != 200):
                return None
            return r.json()


        lottie_logout = load_lottieurl("https://assets6.lottiefiles.com/private_files/lf30_q5liowjw.json")
        st_lottie(
            lottie_logout,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=500,
            width=600,
            key="logout",
        )
    with col1:
        if st.button("LogOut"):
            st.session_state['login'] = False
            st.session_state['user_name'] = ''
            st.header("""  You successfully logged out from bookSpot.""")

if tab == "Search History":
    def load_lottieurl(url: str):
        r = requests.get(url)
        if (r.status_code != 200):
            return None
        return r.json()


    filter_by = {'username': st.session_state['user_name']}
    items_browse_hist_num = search_db.count_documents(filter_by)
    if items_browse_hist_num == 0:
        st.title("This list contain no books.")
        lottie_cart = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kq5rGs.json")
        st_lottie(
            lottie_cart,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=650,
            width=None,
            key="empty_cart"
        )
    else:
        rows_num = 1
        if items_browse_hist_num > 5:
            if items_browse_hist_num % 5 == 0:
                rows_num = int(items_browse_hist_num / 5)
            else:
                rows_num = int(items_browse_hist_num / 5) + 1

        x = search_db.find({'username': st.session_state['user_name']})
        k = items_browse_hist_num - 1
        for i in range(rows_num):
            col = st.columns(5)
            for j in range(5):
                if k >= 0:
                    idx = content_df[content_df['title'] == x[k]["book"]].index[0]
                    col[j].image(content_df.iloc[idx][6], caption=content_df.iloc[idx][1], width=200)
                    k -= 1

# Book Shelf page
if tab == "Book Shelf":
    st.header("Enjoy your favorite Genre of Books🤩")
    st.sidebar.header(""" **Filters** """)

    # it will recommend based on whats your inupt on genre and "sort by"
    def recommend_filter_based(temp_df, sort_by='Most Liked'):
        temp_df.sort_values(by='avg_rating', ascending=False, inplace=True, kind='quicksort')
        temp_df['price'] = temp_df['price'].astype(int)
        if sort_by == 'Popularity':
            temp_df.sort_values(by='rating_num', ascending=False, inplace=True, kind='quicksort')

        elif sort_by == 'Price(high-to-low)':
            temp_df.sort_values(by='price', ascending=False, inplace=True, kind='quicksort')

        elif sort_by == 'Price(low-to-high)':
            temp_df.sort_values(by='price', ascending=True, inplace=True, kind='quicksort')

        final = temp_df.head(10)
        final.reset_index(inplace=True)
        final.drop(['index'], axis=1, inplace=True)
        final['price'] = final['price'].astype(str)
        final['cost'] = "Rs. " + final['price']
        cols = st.columns(5)
        for i in range(5):
            cols[i].text(final.iloc[i][0])
            cols[i].image(final.iloc[i][5], caption=final.iloc[i][8], width=200)
        for i in range(5):
            cols[i].text(final.iloc[i + 5][0])
            cols[i].image(final.iloc[i + 5][5], caption=final.iloc[i + 5][8], width=200)


    def filters():
        genre = st.sidebar.radio('Categories',
                                 options=['Biographies, Memoirs and General Non-Fiction Books', 'Political Books',
                                          'Fantasy Fiction Books', 'Science Fiction Books',
                                          'Crime, Thriller and Suspense Books', 'Children and Young Adult Books',
                                          'Action and Adventure Books'])
        sort = st.sidebar.radio('Sort by',
                                options=['Most liked', 'Popularity', 'Price(high-to-low)', 'Price(low-to-high)'])
        if genre == 'Biographies, Memoirs and General Non-Fiction Books':
            recommend_filter_based(nonfiction_df, sort)
        elif genre == 'Political Books':
            recommend_filter_based(politic_df, sort)
        elif genre == 'Fantasy Fiction Books':
            recommend_filter_based(fantasy_df, sort)
        elif genre == 'Science Fiction Books':
            recommend_filter_based(science_fiction_df, sort)
        elif genre == 'Crime, Thriller and Suspense Books':
            recommend_filter_based(crime_thrill_df, sort)
        elif genre == 'Children and Young Adult Books':
            recommend_filter_based(child_df, sort)
        elif genre == 'Action and Adventure Books':
            recommend_filter_based(axn_adv_df, sort)


    filters()

# Search books
if tab == "Search":

    st.header('Book Recommendation Engine')

    # creating and training the model for content and collaborative based recommendation
    # for content based filtering
    # vectorizing the "tags" of books
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    vector_form = vectorizer.fit_transform(content_df['tags']).toarray()

    sim = cosine_similarity(vector_form)

    # for item to item collaborative based filtering
    # creating a pivot table to feed into knn algorithm
    collab_pivot = collab_df.pivot_table(columns='user_id', index='title', values='rating')
    collab_pivot.fillna(0, inplace=True)
    # creating a sparse matrix to save time complexity because of lots of 0s
    collab_sparse = csr_matrix(collab_pivot)
    # creating and training the model in one step
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5).fit(collab_sparse)

    # scoring the items similar to searched book based on its similarity score and average rating
    # popularity filter
    def scoring(avg_rating, sim_score, rating_count, x, y):
        final_score = sim_score * ((rating_count / (rating_count + y) * avg_rating) + (y / (rating_count + y) * x))
        return final_score

    sim_books = []

    #it will recommend books based on content filtering
    def recommend_content_based(book_name):
        global cart
        book_idx = content_df[content_df['title'] == book_name].index[0]
        # Tim sort which is a hybrid sorting algorithm of merge sort and insertion sort
        book_list = sorted(list(enumerate(sim[book_idx])), reverse=True, key=lambda x: x[1])[1:21]
        indices = []
        for i in book_list:
            indices.append(i[0])
        books_data = content_df.iloc[indices][['isbn', 'title', 'genre', 'avg_rating', 'rating_count', 'poster_url']]
        rating_num = books_data['rating_count']
        rating_avg = books_data['avg_rating']
        x = rating_avg.mean()
        y = rating_num.quantile(0.70)
        books_data['scores'] = " "
        books_data.reset_index(inplace=True)

        # scoring
        for i in range(books_data.shape[0]):
            score = scoring(books_data.iloc[i, 4], book_list[i][1], books_data.iloc[i][5], x, y)
            books_data.at[i, 'scores'] = score

        # sorting
        books_data.sort_values('scores', ascending=False, inplace=True, kind='quicksort')

        books_data.reset_index(drop=True, inplace=True)

        # ranking
        books_data['ranks'] = " "
        for i in range(books_data.shape[0]):
            books_data.at[i, 'ranks'] = i + 1

        # displaying
        mybook = content_df.loc[book_idx]
        final_df = books_data[['isbn', 'title', 'genre', 'scores', 'ranks', 'poster_url']]
        colm1, colm2, col3 = st.columns(3)
        colm1.write(""" ### Your book is: """)
        colm1.image(content_df.iloc[book_idx][6], caption=mybook[1], width=300)
        colm2.write(""" ### About the book:""")
        colm2.write(""" **Title:** """)
        colm2.write(mybook[1])
        colm2.write(""" **ISBN:** """)
        colm2.write(content_df.iloc[book_idx][0])
        colm2.write(""" **Author:** """)
        colm2.write(content_df.iloc[book_idx][3])
        colm2.write(""" **Publisher:** """)
        colm2.write(content_df.iloc[book_idx][5])
        colm2.write(""" **Year of Publication:** """)
        colm2.write(str(content_df.iloc[book_idx][4]))
        sim_books.append(mybook[1])

        with col3:
            if st.button('Add to Cart'):
                if st.session_state["user_name"] == '':
                    st.warning("Please login to add items to the cart!")
                else:
                    record = {
                        'username': st.session_state['user_name'],
                        'book': mybook[1]
                    }
                    if dbCart.count_documents(record) < 1:
                        dbCart.insert_one(record)
            rating = st.number_input("Rate the book?", min_value=0, max_value=10, value=0, step=1)
            if st.button("Update Rating"):
                if st.session_state["user_name"] == '':
                    st.warning("Please login to rate items!")
                else:
                    record = {
                        'username': st.session_state["user_name"],
                        'book': mybook[1],
                        'rating': rating
                    }
                    filterby = {
                        'username': st.session_state["user_name"],
                        'book': mybook[1]
                    }
                    if ratingDB.count_documents(filterby, limit=1) >= 1:
                        ratingDB.update_one(filterby, {"$set": {"rating": rating}})
                    else:
                        ratingDB.insert_one(record)

            if st.text_area("Review the book?"):
                if st.session_state["user_name"] == '':
                    st.warning("Please login to review items!")

        st.write(""" #### Similar books: """)
        cols = st.columns(5)
        for i in range(5):
            sim_books.append(final_df.iloc[i][1])
            cols[i].image(final_df.iloc[i][5], caption=final_df.iloc[i][1], width=200)


    list_book = []
    # it will recommend books based on collaborative filtering
    def recommend_collab_based(book):
        book_index = np.where(collab_pivot.index == book)[0][0]
        distances, indices = knn_model.kneighbors(collab_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=11)
        st.write(""" #### Customers who liked your searched book also liked:""")
        for i in range(11):
            list_book.append(collab_pivot.index[indices[0][i]])

        k = 0
        cols = st.columns(5)
        for i in list_book:
            if any(i in j for j in sim_books):
                continue
            else:
                idx = collab_df[collab_df['title'] == i].index[0]
                cols[k].image(collab_df.iat[idx, 2], caption=i, width=200)
                k = k + 1
                if k > 4:
                    break


    selected_book = st.selectbox('Which book would you like to read?', content_df['title'].values)
    username = st.session_state['user_name']
    record = {
        'username': st.session_state["user_name"],
        'book': selected_book
    }

    if (not st.session_state['user_name'] == '') & (search_db.count_documents(record, limit=1) < 1):
        search_db.insert_one(record)

    recommend_content_based(selected_book)
    recommend_collab_based(selected_book)
