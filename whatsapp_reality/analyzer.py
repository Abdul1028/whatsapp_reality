"""
Core analysis functions for WhatsApp chat data.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
from urlextract import URLExtract
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import os
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Tuple, Dict, Union, Optional

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

extract = URLExtract()

def fetch_stats(selected_user: str, df: pd.DataFrame) -> tuple:
    """
    Fetch basic statistics from the chat data.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        tuple: (num_messages, num_words, num_media_messages, num_links)
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'].str.contains('<Media omitted>|video omitted|image omitted')].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df: pd.DataFrame) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Analyze user activity to find the most active users.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        tuple: (plotly figure, DataFrame with user percentages)
    """
    x = df['user'].value_counts().head()
    df_percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})

    fig = px.bar(x=x.index, y=x.values, labels={'x': 'User', 'y': 'Count'})
    fig.update_layout(title="Most Busy Users")
    fig.update_xaxes(title_text='User', tickangle=-45)
    fig.update_yaxes(title_text='Count')

    return fig, df_percent

def create_wordcloud(selected_user: str, df: pd.DataFrame, stop_words_file: Optional[str] = None) -> WordCloud:
    """
    Create a word cloud from chat messages.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        stop_words_file: Path to file containing stop words
        
    Returns:
        WordCloud object
    """
    if stop_words_file:
        with open(stop_words_file, 'r') as f:
            stop_words = f.read()
    else:
        stop_words = set(STOPWORDS)

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        return " ".join(word for word in message.lower().split() if word not in stop_words)

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user: str, df: pd.DataFrame, stop_words_file: Optional[str] = None) -> go.Figure:
    """
    Analyze and visualize the most common words in chat messages.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        stop_words_file: Path to file containing stop words
        
    Returns:
        plotly Figure object
    """
    if stop_words_file:
        with open(stop_words_file, 'r') as f:
            stop_words = f.read()
    else:
        stop_words = set(STOPWORDS)

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(25), columns=['Word', 'Frequency'])
    fig = px.bar(most_common_df, x='Word', y='Frequency')
    fig.update_layout(title="Most Common Words")
    fig.update_xaxes(title_text='Word', tickangle=-45)
    fig.update_yaxes(title_text='Frequency')
    
    return fig

def emoji_analysis(selected_user: str, df: pd.DataFrame) -> go.Figure:
    """
    Analyze and visualize emoji usage in chat messages.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        plotly Figure object
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.UNICODE_EMOJI['en']])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    
    fig = px.pie(emoji_df.head(8), values=1, names=0, title="Emoji Distribution")
    return fig

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> go.Figure:
    """
    Create a monthly timeline visualization of chat activity.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        plotly Figure object
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = [f"{timeline['month'][i]}-{timeline['year'][i]}" for i in range(timeline.shape[0])]
    timeline['time'] = time

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timeline['time'], y=timeline['message'], mode='lines', marker=dict(color='green')))
    fig.update_layout(title="Monthly Timeline", xaxis_tickangle=-45)
    
    return fig

def daily_timeline(selected_user: str, df: pd.DataFrame) -> go.Figure:
    """
    Create a daily timeline visualization of chat activity.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        plotly Figure object
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_timeline['only_date'], y=daily_timeline['message'], 
                            mode='lines', marker=dict(color='black')))
    fig.update_layout(title="Daily Timeline", xaxis_tickangle=-45)
    
    return fig

def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an activity heatmap showing message patterns by day and hour.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        DataFrame containing the heatmap data
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df.pivot_table(index='day_name', columns='period', 
                         values='message', aggfunc='count').fillna(0)

def analyze_sentiment(message: str) -> str:
    """
    Analyze the sentiment of a message.
    
    Args:
        message: The text message to analyze
        
    Returns:
        str: 'Positive', 'Negative', or 'Neutral'
    """
    blob = TextBlob(message)
    sentiment_score = blob.sentiment.polarity
    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def calculate_sentiment_percentage(selected_users: Union[str, List[str]], df: pd.DataFrame) -> Tuple[Dict, str, str]:
    """
    Calculate sentiment percentages for users.
    
    Args:
        selected_users: User(s) to analyze ('Overall' or list of usernames)
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        tuple: (sentiment_percentages_dict, most_positive_user, most_negative_user)
    """
    if selected_users == 'Overall':
        selected_df = df
    else:
        if isinstance(selected_users, str):
            selected_users = [selected_users]
        selected_df = df[df['user'].isin(selected_users)]

    sid = SentimentIntensityAnalyzer()
    user_sentiment_percentages = {}

    for user, messages in selected_df.groupby('user')['message']:
        positive_count = 0
        negative_count = 0

        for message in messages:
            sentiment_score = sid.polarity_scores(message)['compound']
            if sentiment_score > 0:
                positive_count += 1
            elif sentiment_score < 0:
                negative_count += 1

        total_messages = len(messages)
        positivity_percentage = (positive_count / total_messages) * 100
        negativity_percentage = (negative_count / total_messages) * 100

        user_sentiment_percentages[user] = (f"{positivity_percentage:.2f}%", f"{negativity_percentage:.2f}%")

    most_positive_user = max(user_sentiment_percentages, key=lambda x: float(user_sentiment_percentages[x][0][:-1]))
    most_negative_user = max(user_sentiment_percentages, key=lambda x: float(user_sentiment_percentages[x][1][:-1]))

    return user_sentiment_percentages, most_positive_user, most_negative_user

def analyze_reply_patterns(df: pd.DataFrame) -> Tuple[str, float, str, str]:
    """
    Analyze reply patterns in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        tuple: (user with longest reply time, max reply time in minutes, message, reply)
    """
    user_encoder = OrdinalEncoder()
    df['User Code'] = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))

    message_senders = df['User Code'].values
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(1, -1)[0] != 0
    sender_changed[0] = False
    is_reply = sender_changed & ~df['user'].eq('group_notification')

    df['Is Reply'] = is_reply

    max_reply_times = df.groupby('user')['Reply Time'].max()
    max_reply_user = max_reply_times.idxmax()
    max_reply_time = max_reply_times.max()

    max_reply_message_index = df[df['Reply Time'] == max_reply_time].index[0]
    max_reply_message = df.loc[max_reply_message_index, 'message']
    reply = df.shift(1).loc[max_reply_message_index, 'message']

    return max_reply_user, max_reply_time, max_reply_message, reply

__all__ = [
    'fetch_stats',
    'most_busy_users',
    'create_wordcloud',
    'most_common_words',
    'emoji_analysis',
    'monthly_timeline',
    'daily_timeline',
    'activity_heatmap',
    'analyze_sentiment',
    'calculate_sentiment_percentage',
    'analyze_reply_patterns'
] 