import string
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import praw
import time
import requests

import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Toxic Comment Detection",
    page_icon="@",
    layout="wide",
)

reddit = praw.Reddit(
    client_id="uunNbz6S77OzFX0agFLGHA",
    client_secret="4IQhI0X45Rlwa8hnaSSq6jnrck8Tew",
    username="Stryker13799",
    password="acorigins13799",
    user_agent="MLOPS_A2",
)


st.title("Toxic Comment Detection")

tab1, tab2 = st.tabs(["Inference", "Dashboard"])

with tab1:
    text_input = st.text_input("Enter some text 👇")

    if text_input:
        st.write(
            "Toxicitiy: ",
            eval(
                requests.post(
                    "http://127.0.0.1:5000/predict", json={"text": str(text_input)}
                ).text
            )["prediction"]["toxicity"],
        )


with tab2:
    _subRedditName = "AskReddit"
    _waitTime = 5

    placeholder = st.empty()

    subreddit = reddit.subreddit(_subRedditName)

    while True:
        startTime = time.time()
        comments_list = []
        time_ = []
        for comment in subreddit.stream.comments(skip_existing=True):
            if time.time() - startTime <= _waitTime:
                comments_list.append(comment.body.strip())
                time_.append(time.time())
            else:
                break

        with placeholder.container():
            toxicity = [
                eval(
                    requests.post(
                        "http://127.0.0.1:5000/predict", json={"text": comments}
                    ).text
                )["prediction"]["toxicity"]
                for comments in comments_list
            ]
            new_df1 = pd.DataFrame()
            new_df1["time"] = time_
            new_df1["toxicity"] = toxicity
            fig = px.line(new_df1, x="time", y="toxicity", title="Toxicity over time")

            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
