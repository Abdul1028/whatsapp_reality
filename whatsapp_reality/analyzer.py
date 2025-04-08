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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import random # Added import

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
        emojis.extend([c for c in message if emoji.is_emoji(c)])

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

def analyze_message_types(selected_user: str, df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze different types of messages in the chat.
    
    Args:
        selected_user: The user to analyze or 'Overall' for all users
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict[str, int]: Dictionary with counts of different message types
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    message_types = {
        'Text': 0,
        'Media': 0,
        'Links': 0,
        'Deleted': 0,
        'Stickers': 0,
        'Documents': 0,
        'Audio': 0,
        'Video': 0,
        'Location': 0
    }
    
    # Count different message types
    message_types['Media'] = df[df['message'].str.contains('<Media omitted>|image omitted|video omitted', na=False)].shape[0]
    message_types['Links'] = df[df['message'].apply(lambda x: bool(extract.find_urls(x)))].shape[0]
    message_types['Deleted'] = df[df['message'].str.contains('This message was deleted', na=False)].shape[0]
    message_types['Stickers'] = df[df['message'].str.contains('sticker omitted', na=False)].shape[0]
    message_types['Documents'] = df[df['message'].str.contains('document omitted', na=False)].shape[0]
    message_types['Audio'] = df[df['message'].str.contains('audio omitted', na=False)].shape[0]
    message_types['Video'] = df[df['message'].str.contains('video omitted', na=False)].shape[0]
    message_types['Location'] = df[df['message'].str.contains('location', na=False)].shape[0]
    
    # Text messages are those that aren't any of the above
    message_types['Text'] = df.shape[0] - sum(count for message_type, count in message_types.items() if message_type != 'Text')
    
    return message_types

def create_message_types_chart(message_types: Dict[str, int]) -> go.Figure:
    """
    Create a pie chart visualization of message types.
    
    Args:
        message_types: Dictionary with counts of different message types
        
    Returns:
        plotly Figure object
    """
    labels = list(message_types.keys())
    values = list(message_types.values())
    
    # Filter out types with zero count
    filtered_labels = [label for label, value in zip(labels, values) if value > 0]
    filtered_values = [value for value in values if value > 0]
    
    fig = px.pie(
        names=filtered_labels,
        values=filtered_values,
        title="Message Types Distribution",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    
    return fig

def analyze_conversation_patterns(df: pd.DataFrame) -> Dict[str, Union[float, int]]:
    """
    Analyze conversation patterns in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with conversation pattern metrics
    """
    # Create a copy and handle potential date/index ambiguity *before* sorting
    df_copy = df.copy() # Create copy first
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # Ambiguous: Reset only the 'date' level from the index, discarding it
            df_copy = df_copy.reset_index(level='date', drop=True)
            # Reset any remaining index levels to columns if they exist
            if not isinstance(df_copy.index, pd.RangeIndex): 
                 df_copy = df_copy.reset_index()
        else:
             # Only in index: Reset all levels, making 'date' a column
             df_copy = df_copy.reset_index()

    # Ensure the dataframe is sorted by date
    # Convert to datetime *just in case* before sorting
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values('date') # Apply sort_values to df_copy
    
    # Calculate time differences between consecutive messages
    df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60  # in minutes
    
    # Identify conversation starts (messages after a long gap)
    conversation_threshold = 60  # 60 minutes gap defines a new conversation
    # Use .fillna(True) for the first row's comparison to handle NaT diff
    df_copy['new_conversation'] = (df_copy['time_diff'] > conversation_threshold).fillna(True)
    
    # Count total conversations
    total_conversations = df_copy['new_conversation'].sum()
    
    # Calculate average conversation length (in messages)
    df_copy['conversation_id'] = df_copy['new_conversation'].cumsum()
    conversation_lengths = df_copy.groupby('conversation_id').size()
    # Avoid division by zero if no conversations found
    avg_conversation_length = conversation_lengths.mean() if not conversation_lengths.empty else 0
    
    # Calculate average conversation duration (in minutes)
    conversation_durations = []
    for conv_id, group in df_copy.groupby('conversation_id'): # Use df_copy
        if len(group) > 1:
            duration = (group['date'].max() - group['date'].min()).total_seconds() / 60
            conversation_durations.append(duration)
    
    avg_conversation_duration = np.mean(conversation_durations) if conversation_durations else 0
    
    # Calculate average messages per user per conversation
    messages_per_user = df_copy.groupby(['conversation_id', 'user']).size() # Use df_copy
    avg_messages_per_user = messages_per_user.mean() if not messages_per_user.empty else 0
    
    # Calculate conversation initiators
    # Ensure we only count actual users, not group notifications
    starters_df = df_copy[(df_copy['new_conversation']) & (df_copy['user'] != 'group_notification')] # Use df_copy
    conversation_starters = starters_df.groupby('user').size()
    
    # Prepare results
    results = {
        'total_conversations': total_conversations,
        'avg_conversation_length': avg_conversation_length,
        'avg_conversation_duration_mins': avg_conversation_duration,
        'avg_messages_per_user': avg_messages_per_user,
        'conversation_starters': conversation_starters.to_dict()
    }
    
    return results

def create_conversation_patterns_chart(conversation_patterns: Dict[str, Union[float, int]]) -> Dict[str, go.Figure]:
    """
    Create visualizations for conversation patterns.
    
    Args:
        conversation_patterns: Dictionary with conversation pattern metrics
        
    Returns:
        Dict[str, go.Figure]: Dictionary of plotly figures
    """
    figures = {}
    
    # Create pie chart for conversation starters
    starters = conversation_patterns.get('conversation_starters', {})
    if starters:
        labels = list(starters.keys())
        values = list(starters.values())
        
        fig = px.pie(
            names=labels,
            values=values,
            title="Conversation Initiators",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
        
        figures['conversation_starters'] = fig
    
    # Create bar chart for conversation metrics
    metrics = {
        'Average Messages per Conversation': conversation_patterns.get('avg_conversation_length', 0),
        'Average Conversation Duration (mins)': conversation_patterns.get('avg_conversation_duration_mins', 0),
        'Average Messages per User': conversation_patterns.get('avg_messages_per_user', 0)
    }
    
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        title="Conversation Metrics",
        labels={'x': 'Metric', 'y': 'Value'}
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    figures['conversation_metrics'] = fig
    
    return figures

def analyze_user_interactions(df: pd.DataFrame) -> Tuple[nx.Graph, Dict[str, Dict[str, int]]]:
    """
    Analyze interactions between users in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Tuple[nx.Graph, Dict]: Network graph and interaction counts dictionary
    """
    # Ensure the dataframe is sorted by date
    df = df.sort_index()
    
    # Filter out group notifications
    df = df[df['user'] != 'group_notification']
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (users)
    users = df['user'].unique()
    G.add_nodes_from(users)
    
    # Track interactions between users
    interactions = {user: {other_user: 0 for other_user in users if other_user != user} for user in users}
    
    # Analyze consecutive messages to find interactions
    prev_user = None
    for user in df['user']:
        if prev_user is not None and user != prev_user:
            # Increment interaction count
            interactions[prev_user][user] += 1
            interactions[user][prev_user] += 1
            
            # Add or update edge weight in the graph
            if G.has_edge(prev_user, user):
                G[prev_user][user]['weight'] += 1
            else:
                G.add_edge(prev_user, user, weight=1)
        
        prev_user = user
    
    return G, interactions

def create_user_interaction_graph(G: nx.Graph) -> go.Figure:
    """
    Create a network visualization of user interactions.
    
    Args:
        G: NetworkX graph of user interactions
        
    Returns:
        plotly Figure object
    """
    # Calculate node positions using a layout algorithm
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )
    
    # Color nodes by the number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='User Interaction Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig

def analyze_time_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze messaging patterns across different time periods.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with time-based analysis dataframes
    """
    # Ensure the dataframe has the necessary time columns
    if 'hour' not in df.columns or 'day_name' not in df.columns:
        df['hour'] = df['date'].dt.hour
        df['day_name'] = df['date'].dt.day_name()
    
    # Hourly activity
    hourly_activity = df.groupby('hour').size().reset_index(name='message_count')
    
    # Daily activity
    daily_activity = df.groupby('day_name').size().reset_index(name='message_count')
    # Ensure days are in correct order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_activity['day_order'] = daily_activity['day_name'].apply(lambda x: day_order.index(x))
    daily_activity = daily_activity.sort_values('day_order').drop('day_order', axis=1)
    
    # Monthly activity
    df['month'] = df['date'].dt.month_name()
    df['month_num'] = df['date'].dt.month
    monthly_activity = df.groupby(['month', 'month_num']).size().reset_index(name='message_count')
    monthly_activity = monthly_activity.sort_values('month_num')
    
    # User activity by hour
    user_hourly = df.pivot_table(
        index='hour', 
        columns='user', 
        values='message', 
        aggfunc='count', 
        fill_value=0
    ).reset_index()
    
    # User activity by day
    user_daily = df.pivot_table(
        index='day_name', 
        columns='user', 
        values='message', 
        aggfunc='count', 
        fill_value=0
    ).reset_index()
    
    # Create day order column for sorting
    user_daily['day_order'] = user_daily['day_name'].apply(lambda x: day_order.index(x))
    user_daily = user_daily.sort_values('day_order').drop('day_order', axis=1)
    
    return {
        'hourly_activity': hourly_activity,
        'daily_activity': daily_activity,
        'monthly_activity': monthly_activity,
        'user_hourly': user_hourly,
        'user_daily': user_daily
    }

def analyze_message_length(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, float]]:
    """
    Analyze message length patterns in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with message length analysis data
    """
    # Calculate message length (word count)
    if 'Message Length' not in df.columns:
        df['Message Length'] = df['message'].apply(lambda x: len(x.split()))
    
    # Overall statistics
    avg_length = df['Message Length'].mean()
    max_length = df['Message Length'].max()
    min_length = df['Message Length'].min()
    
    # Message length by user
    user_avg_length = df.groupby('user')['Message Length'].mean().reset_index()
    user_max_length = df.groupby('user')['Message Length'].max().reset_index()
    
    # Find the longest message and its author
    longest_msg_idx = df['Message Length'].idxmax()
    longest_msg = {
        'user': df.loc[longest_msg_idx, 'user'],
        'message': df.loc[longest_msg_idx, 'message'],
        'length': df.loc[longest_msg_idx, 'Message Length'],
        'date': df.loc[longest_msg_idx, 'date']
    }
    
    # Message length distribution
    length_distribution = df['Message Length'].value_counts().sort_index().reset_index()
    length_distribution.columns = ['message_length', 'count']
    
    # Message length over time
    df['date_only'] = df['date'].dt.date
    length_over_time = df.groupby('date_only')['Message Length'].mean().reset_index()
    
    return {
        'avg_length': avg_length,
        'max_length': max_length,
        'min_length': min_length,
        'user_avg_length': user_avg_length,
        'user_max_length': user_max_length,
        'longest_message': longest_msg,
        'length_distribution': length_distribution,
        'length_over_time': length_over_time
    }

def analyze_response_times(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, float]]:
    """
    Analyze response time patterns between users.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with response time analysis data
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Reset index if date is in the index, but handle the case where 'date' column already exists
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, just use the DataFrame as is
            pass
        else:
            # Reset index but don't include the index as a column
            df_copy = df_copy.reset_index(drop=True)
    
    # Ensure the dataframe is sorted by date
    df_copy = df_copy.sort_index()
    
    # Filter out group notifications
    df_copy = df_copy[df_copy['user'] != 'group_notification']
    
    # Calculate time differences between consecutive messages
    df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60  # in minutes
    
    # Identify responses (messages from a different user than the previous message)
    df_copy['prev_user'] = df_copy['user'].shift(1)
    df_copy['is_response'] = (df_copy['user'] != df_copy['prev_user']) & (~df_copy['prev_user'].isna())
    
    # Filter to only include responses
    responses = df_copy[df_copy['is_response']].copy()
    
    # Calculate overall statistics
    avg_response_time = responses['time_diff'].mean()
    median_response_time = responses['time_diff'].median()
    max_response_time = responses['time_diff'].max()
    
    # Calculate response times by user
    user_response_times = responses.groupby('user')['time_diff'].agg(['mean', 'median', 'max']).reset_index()
    
    # Calculate response times between specific user pairs
    user_pairs = []
    for responder in df_copy['user'].unique():
        for original in df_copy['user'].unique():
            if responder != original:
                pair_responses = responses[(responses['user'] == responder) & (responses['prev_user'] == original)]
                if not pair_responses.empty:
                    user_pairs.append({
                        'responder': responder,
                        'original': original,
                        'avg_time': pair_responses['time_diff'].mean(),
                        'count': len(pair_responses)
                    })
    
    user_pair_df = pd.DataFrame(user_pairs)
    
    # Response time distribution
    # Cap at 60 minutes to avoid extreme outliers
    capped_times = responses['time_diff'].clip(upper=60)
    time_distribution = capped_times.value_counts(bins=12).sort_index().reset_index()
    time_distribution.columns = ['time_range', 'count']
    
    # Response time by hour of day
    responses['hour'] = responses['date'].dt.hour
    time_by_hour = responses.groupby('hour')['time_diff'].mean().reset_index()
    
    # Response time by day of week
    responses['day_name'] = responses['date'].dt.day_name()
    time_by_day = responses.groupby('day_name')['time_diff'].mean().reset_index()
    
    # Ensure days are in correct order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    time_by_day['day_order'] = time_by_day['day_name'].apply(lambda x: day_order.index(x))
    time_by_day = time_by_day.sort_values('day_order').drop('day_order', axis=1)
    
    return {
        'avg_response_time': avg_response_time,
        'median_response_time': median_response_time,
        'max_response_time': max_response_time,
        'user_response_times': user_response_times,
        'user_pair_response_times': user_pair_df,
        'time_distribution': time_distribution,
        'time_by_hour': time_by_hour,
        'time_by_day': time_by_day
    }

def analyze_topic_modeling(df: pd.DataFrame, num_topics: int = 5, num_words: int = 10) -> Dict[str, Union[List, pd.DataFrame]]:
    """
    Perform topic modeling on chat messages using Latent Dirichlet Allocation.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        num_topics: Number of topics to extract
        num_words: Number of words per topic to return
        
    Returns:
        Dict: Dictionary with topic modeling results
    """
    # Filter out non-text messages and group notifications
    text_df = df[(~df['message'].str.contains('<Media omitted>|image omitted|video omitted|sticker omitted', na=False)) & 
                 (df['user'] != 'group_notification')].copy()
    
    # Combine all messages into a single corpus
    corpus = text_df['message'].tolist()
    
    # Create a document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,         # Ignore terms that appear in >95% of documents
        min_df=2,            # Ignore terms that appear in <2 documents
        stop_words='english', # Remove English stop words
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only consider words with 3+ characters
    )
    
    # Fit the vectorizer and transform the corpus
    try:
        dtm = vectorizer.fit_transform(corpus)
        
        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()
        
        # Create and fit the LDA model
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method='online'
        )
        
        lda.fit(dtm)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic[top_words_idx].tolist()
            })
        
        # Transform documents to get topic distributions
        doc_topic_dist = lda.transform(dtm)
        
        # Assign dominant topic to each message
        text_df['dominant_topic'] = doc_topic_dist.argmax(axis=1)
        
        # Count messages per topic
        topic_counts = text_df['dominant_topic'].value_counts().sort_index().reset_index()
        topic_counts.columns = ['topic_id', 'message_count']
        
        # Get topic distribution by user
        user_topics = text_df.groupby(['user', 'dominant_topic']).size().reset_index(name='count')
        
        # Calculate topic evolution over time
        text_df['month_year'] = text_df['date'].dt.strftime('%Y-%m')
        topic_evolution = text_df.groupby(['month_year', 'dominant_topic']).size().reset_index(name='count')
        topic_evolution = topic_evolution.sort_values('month_year')
        
        return {
            'topics': topics,
            'topic_counts': topic_counts,
            'user_topics': user_topics,
            'topic_evolution': topic_evolution,
            'success': True
        }
    
    except Exception as e:
        # Return empty results if topic modeling fails
        return {
            'topics': [],
            'topic_counts': pd.DataFrame(),
            'user_topics': pd.DataFrame(),
            'topic_evolution': pd.DataFrame(),
            'success': False,
            'error': str(e)
        }

def analyze_emoji_usage(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Analyze emoji usage patterns in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with emoji usage analysis data
    """
    # Extract all emojis from messages
    emoji_list = []
    for message in df['message']:
        emojis = [c for c in message if emoji.is_emoji(c)]
        emoji_list.extend(emojis)
    
    # Count emoji frequencies
    emoji_counts = Counter(emoji_list)
    
    # Create DataFrame of emoji counts
    emoji_df = pd.DataFrame(emoji_counts.most_common(), columns=['emoji', 'count'])
    
    # Calculate emoji usage by user
    user_emoji = {}
    for user in df['user'].unique():
        user_messages = df[df['user'] == user]['message']
        user_emojis = []
        for message in user_messages:
            user_emojis.extend([c for c in message if emoji.is_emoji(c)])
        
        if user_emojis:
            user_emoji[user] = Counter(user_emojis).most_common(10)
    
    # Calculate emoji usage over time
    df['date_only'] = df['date'].dt.date
    emoji_by_date = {}
    for date, group in df.groupby('date_only'):
        date_emojis = []
        for message in group['message']:
            date_emojis.extend([c for c in message if emoji.is_emoji(c)])
        
        if date_emojis:
            emoji_by_date[date] = Counter(date_emojis).most_common(5)
    
    # Convert emoji_by_date to DataFrame for easier plotting
    emoji_time_data = []
    for date, emoji_counts in emoji_by_date.items():
        for emoji_char, count in emoji_counts:
            emoji_time_data.append({
                'date': date,
                'emoji': emoji_char,
                'count': count
            })
    
    emoji_time_df = pd.DataFrame(emoji_time_data)
    
    # Calculate emoji diversity (number of unique emojis used)
    emoji_diversity = len(emoji_counts)
    
    # Calculate emoji density (emojis per message)
    emoji_density = len(emoji_list) / len(df)
    
    return {
        'emoji_counts': emoji_df,
        'user_emoji': user_emoji,
        'emoji_time_df': emoji_time_df,
        'emoji_diversity': emoji_diversity,
        'emoji_density': emoji_density,
        'total_emojis': len(emoji_list)
    }

def analyze_sentiment_trends(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Analyze sentiment trends over time and by user.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with sentiment analysis data
    """
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each message
    df['sentiment_score'] = df['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    
    # Calculate overall sentiment statistics
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    avg_sentiment = df['sentiment_score'].mean()
    
    # Calculate sentiment by user
    user_sentiment = df.groupby('user')['sentiment_score'].mean().reset_index()
    user_sentiment_counts = df.groupby(['user', 'sentiment']).size().unstack(fill_value=0).reset_index()
    
    # Calculate sentiment over time
    df['date_only'] = df['date'].dt.date
    sentiment_by_date = df.groupby('date_only')['sentiment_score'].mean().reset_index()
    
    # Calculate sentiment by day of week
    df['day_name'] = df['date'].dt.day_name()
    sentiment_by_day = df.groupby('day_name')['sentiment_score'].mean().reset_index()
    
    # Ensure days are in correct order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sentiment_by_day['day_order'] = sentiment_by_day['day_name'].apply(lambda x: day_order.index(x))
    sentiment_by_day = sentiment_by_day.sort_values('day_order').drop('day_order', axis=1)
    
    # Calculate sentiment by hour
    df['hour'] = df['date'].dt.hour
    sentiment_by_hour = df.groupby('hour')['sentiment_score'].mean().reset_index()
    
    # Find most positive and negative messages
    most_positive_idx = df['sentiment_score'].idxmax()
    most_positive = {
        'user': df.loc[most_positive_idx, 'user'],
        'message': df.loc[most_positive_idx, 'message'],
        'score': df.loc[most_positive_idx, 'sentiment_score'],
        'date': df.loc[most_positive_idx, 'date']
    }
    
    most_negative_idx = df['sentiment_score'].idxmin()
    most_negative = {
        'user': df.loc[most_negative_idx, 'user'],
        'message': df.loc[most_negative_idx, 'message'],
        'score': df.loc[most_negative_idx, 'sentiment_score'],
        'date': df.loc[most_negative_idx, 'date']
    }
    
    return {
        'sentiment_counts': sentiment_counts,
        'avg_sentiment': avg_sentiment,
        'user_sentiment': user_sentiment,
        'user_sentiment_counts': user_sentiment_counts,
        'sentiment_by_date': sentiment_by_date,
        'sentiment_by_day': sentiment_by_day,
        'sentiment_by_hour': sentiment_by_hour,
        'most_positive': most_positive,
        'most_negative': most_negative
    }

def analyze_word_usage(df: pd.DataFrame, stop_words: Optional[set] = None) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Analyze word usage patterns in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        stop_words: Set of stop words to exclude from analysis
        
    Returns:
        Dict: Dictionary with word usage analysis data
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Reset index if date is in the index, but handle the case where 'date' column already exists
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, just use the DataFrame as is
            pass
        else:
            # Reset index but don't include the index as a column
            df_copy = df_copy.reset_index(drop=True)
    
    if stop_words is None:
        stop_words = set(STOPWORDS)
    
    # Extract all words from messages
    all_words = []
    for message in df_copy['message']:
        # Skip media messages
        if '<Media omitted>' in message or 'image omitted' in message or 'video omitted' in message:
            continue
        
        # Clean and tokenize
        clean_message = re.sub(r'[^\w\s]', '', message.lower())
        words = clean_message.split()
        
        # Filter stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(filtered_words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Create DataFrame of word counts
    word_df = pd.DataFrame(word_counts.most_common(100), columns=['word', 'count'])
    
    # Calculate word usage by user
    user_words = {}
    for user in df_copy['user'].unique():
        user_messages = df_copy[df_copy['user'] == user]['message']
        user_word_list = []
        
        for message in user_messages:
            if '<Media omitted>' in message or 'image omitted' in message or 'video omitted' in message:
                continue
            
            clean_message = re.sub(r'[^\w\s]', '', message.lower())
            words = clean_message.split()
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            user_word_list.extend(filtered_words)
        
        if user_word_list:
            user_words[user] = Counter(user_word_list).most_common(20)
    
    # Calculate unique words per user
    user_unique_words = {}
    for user, word_counts in user_words.items():
        user_unique_words[user] = len(set([word for word, count in word_counts]))
    
    # Calculate word diversity (number of unique words used)
    word_diversity = len(word_counts)
    
    # Calculate average words per message
    words_per_message = len(all_words) / len(df_copy)
    
    return {
        'word_counts': word_df,
        'user_words': user_words,
        'user_unique_words': user_unique_words,
        'word_diversity': word_diversity,
        'words_per_message': words_per_message,
        'total_words': len(all_words)
    }

def analyze_conversation_flow(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Analyze conversation flow patterns in the chat.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with conversation flow analysis data
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Reset index if date is in the index, but handle the case where 'date' column already exists
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, just use the DataFrame as is
            pass
        else:
            # Reset index but don't include the index as a column
            df_copy = df_copy.reset_index(drop=True)
    
    # Ensure the dataframe is sorted by date
    df_copy = df_copy.sort_index()
    
    # Calculate time differences between consecutive messages
    df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60  # in minutes
    
    # Identify conversation starts (messages after a long gap)
    conversation_threshold = 60  # 60 minutes gap defines a new conversation
    df_copy['new_conversation'] = df_copy['time_diff'] > conversation_threshold
    df_copy['conversation_id'] = df_copy['new_conversation'].cumsum()
    
    # Calculate conversation statistics
    conversation_stats = df_copy.groupby('conversation_id').agg(
        start_time=('date', 'min'),
        end_time=('date', 'max'),
        duration=('date', lambda x: (x.max() - x.min()).total_seconds() / 60),
        message_count=('message', 'count'),
        participants=('user', lambda x: len(x.unique()))
    ).reset_index()
    
    # Calculate user participation in conversations
    user_participation = df_copy.groupby(['conversation_id', 'user']).size().reset_index(name='message_count')
    
    # Calculate conversation starters
    conversation_starters = df_copy[df_copy['new_conversation']].groupby('user').size().reset_index(name='count')
    
    # Calculate conversation enders (last message in each conversation)
    conversation_enders = df_copy.groupby('conversation_id').tail(1).groupby('user').size().reset_index(name='count')
    
    # Calculate conversation activity by hour
    df_copy['hour'] = df_copy['date'].dt.hour
    conversation_by_hour = df_copy.groupby(['conversation_id', 'hour']).size().reset_index(name='message_count')
    hour_conversation_counts = conversation_by_hour.groupby('hour').size().reset_index(name='conversation_count')
    
    # Calculate conversation activity by day
    df_copy['day_name'] = df_copy['date'].dt.day_name()
    conversation_by_day = df_copy.groupby(['conversation_id', 'day_name']).size().reset_index(name='message_count')
    day_conversation_counts = conversation_by_day.groupby('day_name').size().reset_index(name='conversation_count')
    
    # Ensure days are in correct order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_conversation_counts['day_order'] = day_conversation_counts['day_name'].apply(lambda x: day_order.index(x))
    day_conversation_counts = day_conversation_counts.sort_values('day_order').drop('day_order', axis=1)
    
    # Calculate conversation density (messages per minute)
    conversation_stats['message_density'] = conversation_stats.apply(
        lambda x: x['message_count'] / x['duration'] if x['duration'] > 0 else 0, axis=1
    )
    
    return {
        'conversation_stats': conversation_stats,
        'user_participation': user_participation,
        'conversation_starters': conversation_starters,
        'conversation_enders': conversation_enders,
        'hour_conversation_counts': hour_conversation_counts,
        'day_conversation_counts': day_conversation_counts,
        'total_conversations': len(conversation_stats)
    }

def analyze_user_activity_patterns(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Analyze detailed activity patterns for each user.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with user activity pattern analysis data
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()

    # Reset index if date is in the index, but handle the case where 'date' column already exists
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, just use the DataFrame as is
            pass
        else:
            # Reset index but don't include the index as a column
            df_copy = df_copy.reset_index(drop=True)

    # Ensure necessary time columns exist
    if 'hour' not in df_copy.columns:
        df_copy['hour'] = df_copy['date'].dt.hour
    if 'day_name' not in df_copy.columns:
        df_copy['day_name'] = df_copy['date'].dt.day_name()
    if 'month' not in df_copy.columns:
        df_copy['month'] = df_copy['date'].dt.month_name()
    
    # Activity patterns by user
    user_patterns = {}
    
    for user in df_copy['user'].unique():
        if user == 'group_notification':
            continue
            
        user_df = df_copy[df_copy['user'] == user]
        
        # Active hours
        active_hours = user_df.groupby('hour').size().reset_index(name='count')
        peak_hour = active_hours.loc[active_hours['count'].idxmax(), 'hour']
        
        # Active days
        active_days = user_df.groupby('day_name').size().reset_index(name='count')
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        active_days['day_order'] = active_days['day_name'].apply(lambda x: day_order.index(x))
        active_days = active_days.sort_values('day_order')
        peak_day = active_days.loc[active_days['count'].idxmax(), 'day_name']
        
        # Active months
        active_months = user_df.groupby('month').size().reset_index(name='count')
        
        # Message frequency
        total_messages = len(user_df)
        messages_per_day = user_df.groupby(user_df['date'].dt.date).size().mean()
        
        # Response patterns
        user_df['prev_user'] = user_df['user'].shift(1)
        user_df['time_diff'] = user_df['date'].diff().dt.total_seconds() / 60
        
        # Who they respond to most
        responds_to = user_df[user_df['prev_user'] != user].groupby('prev_user').size()
        if not responds_to.empty:
            most_responds_to = responds_to.idxmax()
            response_count = responds_to.max()
        else:
            most_responds_to = None
            response_count = 0
        
        # Who responds to them most
        other_users_df = df_copy[df_copy['prev_user'] == user]
        responded_by = other_users_df.groupby('user').size()
        if not responded_by.empty:
            most_responded_by = responded_by.idxmax()
            responded_by_count = responded_by.max()
        else:
            most_responded_by = None
            responded_by_count = 0
        
        # Activity streaks
        user_df['date_only'] = user_df['date'].dt.date
        daily_counts = user_df.groupby('date_only').size()
        active_days_count = len(daily_counts)
        
        # Find longest streak of consecutive days
        dates_array = sorted(daily_counts.index)
        if dates_array:
            streaks = []
            current_streak = [dates_array[0]]
            
            for i in range(1, len(dates_array)):
                if (dates_array[i] - dates_array[i-1]).days == 1:
                    current_streak.append(dates_array[i])
                else:
                    streaks.append(current_streak)
                    current_streak = [dates_array[i]]
            
            streaks.append(current_streak)
            longest_streak = max(streaks, key=len)
            longest_streak_length = len(longest_streak)
            longest_streak_start = longest_streak[0]
            longest_streak_end = longest_streak[-1]
        else:
            longest_streak_length = 0
            longest_streak_start = None
            longest_streak_end = None
        
        # Store user patterns
        user_patterns[user] = {
            'total_messages': total_messages,
            'messages_per_day': messages_per_day,
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'active_hours': active_hours,
            'active_days': active_days,
            'active_months': active_months,
            'most_responds_to': most_responds_to,
            'response_count': response_count,
            'most_responded_by': most_responded_by,
            'responded_by_count': responded_by_count,
            'active_days_count': active_days_count,
            'longest_streak_length': longest_streak_length,
            'longest_streak_start': longest_streak_start,
            'longest_streak_end': longest_streak_end
        }
    
    return {
        'user_patterns': user_patterns
    }

def analyze_conversation_mood_shifts(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, List, Dict]]:
    """
    Analyze how conversation mood shifts throughout chats.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with mood shift analysis data
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Reset index if date is in the index, but handle the case where 'date' column already exists
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, just use the DataFrame as is
            pass
        else:
            # Reset index but don't include the index as a column
            df_copy = df_copy.reset_index(drop=True)
    
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each message
    df_copy['sentiment_score'] = df_copy['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # Ensure the dataframe is sorted by date
    df_copy = df_copy.sort_index()
    
    # Identify conversation starts (messages after a long gap)
    if 'time_diff' not in df_copy.columns:
        df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60  # in minutes
    
    conversation_threshold = 60  # 60 minutes gap defines a new conversation
    df_copy['new_conversation'] = df_copy['time_diff'] > conversation_threshold
    df_copy['conversation_id'] = df_copy['new_conversation'].cumsum()
    
    # Calculate mood shifts within conversations
    df_copy['prev_sentiment'] = df_copy['sentiment_score'].shift(1)
    df_copy['sentiment_shift'] = df_copy['sentiment_score'] - df_copy['prev_sentiment']
    
    # Only consider shifts between different users
    df_copy['prev_user'] = df_copy['user'].shift(1)
    df_copy['is_response'] = (df_copy['user'] != df_copy['prev_user']) & (~df_copy['prev_user'].isna())
    
    # Filter to only include responses that cause significant mood shifts
    significant_shift = 0.5  # Threshold for significant mood shift
    mood_lifters = df_copy[(df_copy['is_response']) & (df_copy['sentiment_shift'] > significant_shift)]
    mood_dampeners = df_copy[(df_copy['is_response']) & (df_copy['sentiment_shift'] < -significant_shift)]
    
    # Identify top mood lifters
    top_lifters = []
    for user in df_copy['user'].unique():
        if user == 'group_notification':
            continue
            
        user_lifts = mood_lifters[mood_lifters['user'] == user]
        if len(user_lifts) > 0:
            total_responses = df_copy[(df_copy['user'] == user) & (df_copy['is_response'])].shape[0]
            if total_responses > 0:
                lift_ratio = len(user_lifts) / total_responses
                top_lifters.append((user, len(user_lifts), lift_ratio))
    
    top_lifters.sort(key=lambda x: x[1], reverse=True)
    
    # Identify top mood dampeners
    top_dampeners = []
    for user in df_copy['user'].unique():
        if user == 'group_notification':
            continue
            
        user_dampens = mood_dampeners[mood_dampeners['user'] == user]
        if len(user_dampens) > 0:
            total_responses = df_copy[(df_copy['user'] == user) & (df_copy['is_response'])].shape[0]
            if total_responses > 0:
                dampen_ratio = len(user_dampens) / total_responses
                top_dampeners.append((user, len(user_dampens), dampen_ratio))
    
    top_dampeners.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate overall mood shifters (both positive and negative)
    top_shifters = []
    for user in df_copy['user'].unique():
        if user == 'group_notification':
            continue
            
        user_shifts = df_copy[(df_copy['user'] == user) & (df_copy['is_response']) & 
                         (abs(df_copy['sentiment_shift']) > significant_shift)]
        if len(user_shifts) > 0:
            total_responses = df_copy[(df_copy['user'] == user) & (df_copy['is_response'])].shape[0]
            if total_responses > 0:
                shift_ratio = len(user_shifts) / total_responses
                avg_shift = user_shifts['sentiment_shift'].mean()
                top_shifters.append((user, len(user_shifts), shift_ratio, avg_shift))
    
    top_shifters.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate conversation mood trajectories
    conversation_moods = {}
    for conv_id, group in df_copy.groupby('conversation_id'):
        if len(group) > 3:  # Only consider conversations with at least 3 messages
            mood_trajectory = group['sentiment_score'].tolist()
            start_mood = mood_trajectory[0]
            end_mood = mood_trajectory[-1]
            mood_change = end_mood - start_mood
            
            conversation_moods[conv_id] = {
                'start_time': group['date'].min(),
                'end_time': group['date'].max(),
                'message_count': len(group),
                'participants': group['user'].nunique(),
                'start_mood': start_mood,
                'end_mood': end_mood,
                'mood_change': mood_change,
                'mood_trajectory': mood_trajectory
            }
    
    # Find conversations with most dramatic mood changes
    dramatic_convs = sorted(
        [(conv_id, data['mood_change']) for conv_id, data in conversation_moods.items()],
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]
    
    return {
        'mood_lifters': top_lifters,
        'mood_dampeners': top_dampeners,
        'top_shifters': top_shifters,
        'conversation_moods': conversation_moods,
        'dramatic_conversations': dramatic_convs
    }

def analyze_conversation_compatibility(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, List, Dict]]:
    """
    Analyze compatibility between users based on their conversation patterns.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with compatibility analysis data
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Reset index if date is in the index, but handle the case where 'date' column already exists
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, just use the DataFrame as is
            pass
        else:
            # Reset index but don't include the index as a column
            df_copy = df_copy.reset_index(drop=True)
    
    # Ensure the dataframe is sorted by date
    df_copy = df_copy.sort_index()
    
    # Filter out group notifications
    df_copy = df_copy[df_copy['user'] != 'group_notification']
    
    # Calculate time differences between consecutive messages
    if 'time_diff' not in df_copy.columns:
        df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60  # in minutes
    
    # Identify responses
    if 'prev_user' not in df_copy.columns:
        df_copy['prev_user'] = df_copy['user'].shift(1)
    
    df_copy['is_response'] = (df_copy['user'] != df_copy['prev_user']) & (~df_copy['prev_user'].isna())
    
    # Calculate sentiment if not already done
    if 'sentiment_score' not in df_copy.columns:
        sid = SentimentIntensityAnalyzer()
        df_copy['sentiment_score'] = df_copy['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # Analyze user pairs
    users = df_copy['user'].unique()
    user_pairs = {}
    
    for i, user1 in enumerate(users):
        for user2 in users[i+1:]:
            pair_key = f"{user1}_{user2}" if user1 < user2 else f"{user2}_{user1}"
            
            # Direct replies between the two users
            user1_to_user2 = df_copy[(df_copy['user'] == user1) & (df_copy['prev_user'] == user2) & (df_copy['is_response'])]
            user2_to_user1 = df_copy[(df_copy['user'] == user2) & (df_copy['prev_user'] == user1) & (df_copy['is_response'])]
            
            direct_replies = len(user1_to_user2) + len(user2_to_user1)
            
            # Skip pairs with too few interactions
            if direct_replies < 5:
                continue
            
            # Calculate average response times
            avg_response_time_1to2 = user1_to_user2['time_diff'].mean() if len(user1_to_user2) > 0 else 0
            avg_response_time_2to1 = user2_to_user1['time_diff'].mean() if len(user2_to_user1) > 0 else 0
            
            # Calculate response rate (percentage of messages that get a response)
            user1_messages = df_copy[df_copy['user'] == user1].shape[0]
            user2_messages = df_copy[df_copy['user'] == user2].shape[0]
            
            response_rate_1to2 = len(user2_to_user1) / user1_messages if user1_messages > 0 else 0
            response_rate_2to1 = len(user1_to_user2) / user2_messages if user2_messages > 0 else 0
            
            # Calculate sentiment alignment
            user1_sentiment = df_copy[df_copy['user'] == user1]['sentiment_score'].mean()
            user2_sentiment = df_copy[df_copy['user'] == user2]['sentiment_score'].mean()
            
            sentiment_diff = abs(user1_sentiment - user2_sentiment)
            sentiment_alignment = 1 - min(sentiment_diff, 1)  # 0 to 1 scale
            
            # Calculate conversation engagement
            # (how often they talk to each other vs. others)
            user1_to_user2_ratio = len(user1_to_user2) / user1_messages if user1_messages > 0 else 0
            user2_to_user1_ratio = len(user2_to_user1) / user2_messages if user2_messages > 0 else 0
            
            engagement_score = (user1_to_user2_ratio + user2_to_user1_ratio) / 2
            
            # Calculate overall compatibility score
            response_time_factor = 1 / (1 + min((avg_response_time_1to2 + avg_response_time_2to1) / 2, 60) / 60)
            response_rate_factor = (response_rate_1to2 + response_rate_2to1) / 2
            
            compatibility_score = (
                0.3 * response_time_factor +
                0.3 * response_rate_factor +
                0.2 * sentiment_alignment +
                0.2 * engagement_score
            ) * 100  # Scale to 0-100
            
            user_pairs[pair_key] = {
                'user1': user1,
                'user2': user2,
                'direct_replies': direct_replies,
                'avg_response_time_1to2': avg_response_time_1to2,
                'avg_response_time_2to1': avg_response_time_2to1,
                'response_rate_1to2': response_rate_1to2,
                'response_rate_2to1': response_rate_2to1,
                'sentiment_alignment': sentiment_alignment,
                'engagement_score': engagement_score,
                'compatibility_score': compatibility_score
            }
    
    # Find most compatible pairs
    most_compatible = sorted(
        [(data['user1'], data['user2'], data['compatibility_score']) 
         for pair_key, data in user_pairs.items()],
        key=lambda x: x[2],
        reverse=True
    )
    
    return {
        'user_pairs': user_pairs,
        'most_compatible': most_compatible
    }

def analyze_personality(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze personality traits of users based on their messaging patterns.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        
    Returns:
        Dict: Dictionary with personality analysis data for each user
    """
    # Filter out group notifications
    df = df[df['user'] != 'group_notification']
    
    # Calculate message length if not already done
    if 'Message Length' not in df.columns:
        df['Message Length'] = df['message'].apply(lambda x: len(x.split()))
    
    # Calculate sentiment if not already done
    if 'sentiment_score' not in df.columns:
        sid = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    
    # Ensure time columns exist
    if 'hour' not in df.columns:
        df['hour'] = df['date'].dt.hour
    
    # Personality traits for each user
    personality_traits = {}
    
    for user in df['user'].unique():
        user_df = df[df['user'] == user]
        
        # Skip users with too few messages
        if len(user_df) < 10:
            continue
        
        # Calculate traits
        
        # 1. Openness (creativity, curiosity, openness to new experiences)
        # - Vocabulary diversity
        # - Message complexity
        # - Time of day variety
        
        # Vocabulary diversity
        user_words = []
        for message in user_df['message']:
            words = message.lower().split()
            user_words.extend([word for word in words if len(word) > 2])
        
        unique_words = len(set(user_words))
        total_words = len(user_words)
        
        vocab_diversity = min(unique_words / max(total_words, 1) * 100, 100) if total_words > 0 else 0
        
        # Message complexity
        avg_message_length = user_df['Message Length'].mean()
        max_message_length = user_df['Message Length'].max()
        
        # Normalize to 0-100 scale
        message_complexity = min(avg_message_length / 10 * 100, 100)
        
        # Time of day variety
        hour_counts = user_df.groupby('hour').size()
        hour_entropy = -(hour_counts / len(user_df) * np.log2(hour_counts / len(user_df))).sum()
        time_variety = min(hour_entropy / 3 * 100, 100)  # Normalize
        
        openness = (vocab_diversity * 0.4 + message_complexity * 0.4 + time_variety * 0.2)
        
        # 2. Conscientiousness (organization, responsibility, reliability)
        # - Response time
        # - Consistency in messaging
        # - Message structure
        
        # Response time
        user_df['prev_user'] = user_df['user'].shift(1)
        user_df['time_diff'] = user_df['date'].diff().dt.total_seconds() / 60
        
        responses = user_df[user_df['prev_user'] != user]
        avg_response_time = responses['time_diff'].mean() if len(responses) > 0 else 60
        
        response_time_score = max(100 - min(avg_response_time, 60), 0)
        
        # Consistency in messaging
        daily_counts = user_df.groupby(user_df['date'].dt.date).size()
        consistency = 100 - min(daily_counts.std() / max(daily_counts.mean(), 1) * 100, 100)
        
        # Message structure (punctuation, capitalization)
        def message_structure_score(message):
            has_proper_capitalization = any(c.isupper() for c in message) and not message.isupper()
            has_punctuation = any(c in '.!?,;:' for c in message)
            return (has_proper_capitalization + has_punctuation) / 2
        
        structure_scores = user_df['message'].apply(message_structure_score)
        avg_structure = structure_scores.mean() * 100
        
        conscientiousness = (response_time_score * 0.4 + consistency * 0.3 + avg_structure * 0.3)
        
        # 3. Extraversion (sociability, assertiveness, energy)
        # - Message frequency
        # - Conversation initiation
        # - Emoji usage
        
        # Message frequency
        messages_per_day = len(user_df) / len(daily_counts) if len(daily_counts) > 0 else 0
        frequency_score = min(messages_per_day / 10 * 100, 100)
        
        # Conversation initiation
        if 'new_conversation' not in df_copy.columns:
            df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60
            df_copy['new_conversation'] = df_copy['time_diff'] > 60
        
        initiations = df_copy[(df_copy['user'] == user) & (df_copy['new_conversation'])].shape[0]
        initiation_ratio = initiations / len(user_df) if len(user_df) > 0 else 0
        initiation_score = min(initiation_ratio * 200, 100)  # Scale up for better distribution
        
        # Emoji usage
        emoji_count = sum(len([c for c in message if emoji.is_emoji(c)]) for message in user_df['message'])
        emoji_per_message = emoji_count / len(user_df) if len(user_df) > 0 else 0
        emoji_score = min(emoji_per_message * 50, 100)  # Scale up for better distribution
        
        extraversion = (frequency_score * 0.4 + initiation_score * 0.4 + emoji_score * 0.2)
        
        # 4. Agreeableness (cooperation, compassion, consideration)
        # - Sentiment positivity
        # - Response rate
        # - Question asking
        
        # Sentiment positivity
        avg_sentiment = user_df['sentiment_score'].mean()
        sentiment_score = (avg_sentiment + 1) / 2 * 100  # Convert from -1,1 to 0-100
        
        # Response rate
        user_messages = df_copy[df_copy['user'] == user].shape[0]
        user_responses = df_copy[(df_copy['prev_user'] == user) & (df_copy['user'] != user)].shape[0]
        
        response_rate = user_responses / user_messages if user_messages > 0 else 0
        response_rate_score = min(response_rate * 150, 100)  # Scale up for better distribution
        
        # Question asking (indicates interest in others)
        question_count = sum('?' in message for message in user_df['message'])
        question_ratio = question_count / len(user_df) if len(user_df) > 0 else 0
        question_score = min(question_ratio * 200, 100)  # Scale up for better distribution
        
        agreeableness = (sentiment_score * 0.4 + response_rate_score * 0.4 + question_score * 0.2)
        
        # 5. Neuroticism (emotional instability, anxiety, negative emotions)
        # - Sentiment volatility
        # - Message editing/deletion
        # - Late night messaging
        
        # Sentiment volatility
        sentiment_std = user_df['sentiment_score'].std()
        volatility_score = min(sentiment_std * 100, 100)
        
        # Message editing/deletion
        deletion_count = sum('This message was deleted' in message for message in user_df['message'])
        deletion_ratio = deletion_count / len(user_df) if len(user_df) > 0 else 0
        deletion_score = min(deletion_ratio * 300, 100)  # Scale up for better distribution
        
        # Late night messaging (10 PM - 4 AM)
        late_night = user_df[(user_df['hour'] >= 22) | (user_df['hour'] <= 4)].shape[0]
        late_night_ratio = late_night / len(user_df) if len(user_df) > 0 else 0
        late_night_score = min(late_night_ratio * 200, 100)  # Scale up for better distribution
        
        neuroticism = (volatility_score * 0.4 + deletion_score * 0.3 + late_night_score * 0.3)
        
        # Store personality traits
        personality_traits[user] = {
            'openness': round(openness),
            'conscientiousness': round(conscientiousness),
            'extraversion': round(extraversion),
            'agreeableness': round(agreeableness),
            'neuroticism': round(neuroticism)
        }
    
    return personality_traits

def predict_future_activity(df: pd.DataFrame, forecast_days: int = 30) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Predict future messaging activity based on historical patterns.
    
    Args:
        df: The preprocessed DataFrame containing chat data
        forecast_days: Number of days to forecast
        
    Returns:
        Dict: Dictionary with prediction data and metrics
    """
    # Check if we have enough data for prediction
    if len(df) < 30:
        return {
            'success': False,
            'error': "Not enough data for prediction. Need at least 30 messages."
        }
    
    # Prepare daily message counts
    df['date_only'] = df['date'].dt.date
    daily_counts = df.groupby('date_only').size().reset_index()
    daily_counts.columns = ['date', 'message_count']
    
    # Fill in missing dates with zero counts
    date_range = pd.date_range(start=daily_counts['date'].min(), end=daily_counts['date'].max())
    date_df = pd.DataFrame({'date': date_range})
    daily_counts = pd.merge(date_df, daily_counts, on='date', how='left').fillna(0)
    
    # Create features for prediction
    daily_counts['day_of_week'] = daily_counts['date'].dt.dayofweek
    daily_counts['month'] = daily_counts['date'].dt.month
    daily_counts['day'] = daily_counts['date'].dt.day
    
    # Create lag features (previous days' counts)
    for lag in range(1, 8):
        daily_counts[f'lag_{lag}'] = daily_counts['message_count'].shift(lag)
    
    # Create rolling average features
    for window in [3, 7, 14]:
        daily_counts[f'rolling_{window}'] = daily_counts['message_count'].rolling(window=window).mean()
    
    # Drop rows with NaN values (from lag and rolling features)
    daily_counts = daily_counts.dropna()
    
    # If we have too little data after creating features, return error
    if len(daily_counts) < 14:
        return {
            'success': False,
            'error': "Not enough data for prediction after creating features."
        }
    
    # Try different models for prediction
    try:
        # Simple time series prediction using historical patterns
        # 1. Calculate average by day of week
        dow_avg = daily_counts.groupby('day_of_week')['message_count'].mean()
        
        # 2. Calculate recent trend
        recent_data = daily_counts.tail(14)
        recent_avg = recent_data['message_count'].mean()
        older_data = daily_counts.iloc[-28:-14]
        older_avg = older_data['message_count'].mean()
        
        trend_factor = recent_avg / older_avg if older_avg > 0 else 1
        trend_factor = max(0.5, min(trend_factor, 1.5))  # Limit trend factor
        
        # 3. Generate future dates
        last_date = daily_counts['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # 4. Predict message counts
        predictions = []
        for future_date in future_dates:
            dow = future_date.dayofweek
            base_prediction = dow_avg[dow]
            adjusted_prediction = base_prediction * trend_factor
            predictions.append({
                'date': future_date.date(),
                'day_of_week': dow,
                'predicted_count': round(adjusted_prediction, 1)
            })
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate prediction metrics
        total_predicted = predictions_df['predicted_count'].sum()
        daily_avg_predicted = predictions_df['predicted_count'].mean()
        max_day_predicted = predictions_df.loc[predictions_df['predicted_count'].idxmax()]
        min_day_predicted = predictions_df.loc[predictions_df['predicted_count'].idxmin()]
        
        # Calculate day of week patterns
        dow_predictions = predictions_df.groupby('day_of_week')['predicted_count'].mean()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_patterns = {day_names[i]: round(dow_predictions[i], 1) for i in range(7)}
        
        # Compare with historical data
        historical_daily_avg = daily_counts['message_count'].mean()
        percent_change = ((daily_avg_predicted / historical_daily_avg) - 1) * 100 if historical_daily_avg > 0 else 0
        
        # Prepare results
        results = {
            'success': True,
            'predictions': predictions_df,
            'metrics': {
                'total_predicted_messages': round(total_predicted),
                'daily_avg_predicted': round(daily_avg_predicted, 1),
                'max_day': {
                    'date': max_day_predicted['date'],
                    'count': round(max_day_predicted['predicted_count'])
                },
                'min_day': {
                    'date': min_day_predicted['date'],
                    'count': round(min_day_predicted['predicted_count'])
                },
                'day_of_week_patterns': dow_patterns,
                'historical_daily_avg': round(historical_daily_avg, 1),
                'percent_change': round(percent_change, 1)
            }
        }
        
        return results
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Error during prediction: {str(e)}"
        }

def create_plotly_wordcloud(selected_user: str, df: pd.DataFrame, include_hinglish_stopwords: bool = False) -> go.Figure:
    """
    Enhanced word cloud generation using Plotly.

    Args:
        selected_user: The user to analyze or 'Overall' for all users.
        df: The preprocessed DataFrame containing chat data.
        include_hinglish_stopwords: 
            If True, attempts to load 'stop_hinglish.txt' from the package directory.
            If the file is found, its contents are used as Hinglish stopwords.
            If the file is NOT found, a predefined sample set of Hinglish stopwords is used as a fallback.
            Defaults to False (no Hinglish stopwords included).

    Returns:
        plotly Figure object representing the word cloud.
    """
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # --- Define Stopwords --- 
    # Standard chat-specific stopwords
    chat_stopwords = {
        'media', 'omitted', 'image', 'video', 'audio', 'sticker', 'gif',
        'http', 'https', 'www', 'com', 'message', 'deleted', 'ok', 'okay',
        'yes', 'no', 'hi', 'hello', 'hey', 'hmm', 'haha', 'lol', 'lmao',
        'thanks', 'thank', 'you', 'the', 'and', 'for', 'this', 'that',
        'have', 'has', 'had', 'not', 'with', 'from', 'your', 'which',
        'there', 'their', 'they', 'them', 'then', 'than', 'but', 'also'
    }
    
    # Predefined Hinglish stopwords sample (used as fallback)
    hinglish_stopwords_sample = {
        'aaj', 'ab', 'abhi', 'acha', 'aur', 'bas', 'bhai', 'bhi', 'bht', 'bol', 
        'hai', 'hain', 'han', 'ho', 'hota', 'hua', 'hui', 'hum', 'hoga', 'hongi',
        'jab', 'kaha', 'kaise', 'kar', 'karo', 'karte', 'ke', 'ki', 'ko', 'koi',
        'kuch', 'kya', 'kyu', 'kyuki', 'mein', 'mera', 'mere', 'meri', 'mujhe', 
        'na', 'nahi', 'nhi', 'par', 'pe', 'phir', 'raha', 'rahe', 'rahi', 'rha',
        'rhe', 'rhi', 'sab', 'se', 'so', 'tab', 'tak', 'tha', 'the', 'thi', 'to',
        'tu', 'tum', 'tera', 'tere', 'teri', 'wo', 'woh', 'ya', 'yah', 'yahi', 'yeh'
    }

    # Combine stopwords based on the parameter and file availability
    all_stopwords = chat_stopwords.copy()
    if include_hinglish_stopwords:
        loaded_from_file = False
        try:
            # Construct path to the stopword file relative to this script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stop_file_path = os.path.join(current_dir, 'stop_hinglish.txt')

            if os.path.exists(stop_file_path):
                with open(stop_file_path, 'r', encoding='utf-8') as f:
                    loaded_hinglish_stopwords = set(word.strip() for word in f.readlines())
                all_stopwords.update(loaded_hinglish_stopwords)
                print(f"Successfully loaded and included {len(loaded_hinglish_stopwords)} Hinglish stopwords from: {stop_file_path}")
                loaded_from_file = True
            else:
                 print(f"Warning: Hinglish stopword file not found at {stop_file_path}.")

        except Exception as e:
            print(f"Error loading stopwords from file '{stop_file_path}': {e}")

        # Fallback to sample list if file wasn't loaded
        if not loaded_from_file:
            all_stopwords.update(hinglish_stopwords_sample)
            print(f"Using fallback sample list of {len(hinglish_stopwords_sample)} Hinglish stopwords.")
            
    else:
        print("Hinglish stopwords were not requested (include_hinglish_stopwords=False).")
        
    # --- Process Messages --- 
    words = []
    for message in df['message']:
        # Skip media messages
        if 'omitted' in message.lower():
            continue
            
        # Clean and tokenize
        clean_message = re.sub(r'[^\w\s]', '', message.lower())
        message_words = clean_message.split()
        
        # Filter stopwords, short words, and words containing digits
        filtered_words = []
        for word in message_words:
            # Skip if word is in stopwords
            if word.lower() in all_stopwords:
                continue
                
            # Skip if word is too short
            if len(word) <= 2:
                continue
                
            # Skip if word contains any digits
            if any(char.isdigit() for char in word):
                continue
                
            filtered_words.append(word)
            
        words.extend(filtered_words)
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    
    # Get top words
    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
    print(f"Top words after filtering: {top_words[:10]}...")
    
    # Prepare data for word cloud
    word_cloud_data = pd.DataFrame(top_words, columns=['word', 'count'])
    
    # Check if we have any words to display
    if word_cloud_data.empty:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No significant words found after filtering stopwords",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Scale sizes for better visualization
    max_count = word_cloud_data['count'].max()
    word_cloud_data['size'] = (word_cloud_data['count'] / max_count * 50) + 10
    
    # Generate random positions with collision detection
    positions = []
    for _ in range(len(word_cloud_data)):
        attempts = 0
        while attempts < 100:  # Limit attempts to avoid infinite loop
            x = random.uniform(-0.8, 0.8)
            y = random.uniform(-0.8, 0.8)
            
            # Check for collisions with existing positions
            collision = False
            for pos in positions:
                # Simple distance check
                if ((x - pos[0])**2 + (y - pos[1])**2) < 0.02:  # Adjust this value for spacing
                    collision = True
                    break
            
            if not collision:
                positions.append((x, y))
                break
            
            attempts += 1
            
        # If we couldn't find a non-colliding position, just add one
        if attempts >= 100:
            positions.append((random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)))
    
    word_cloud_data['x'] = [pos[0] for pos in positions]
    word_cloud_data['y'] = [pos[1] for pos in positions]
    
    # Create color scale based on frequency
    word_cloud_data['color'] = word_cloud_data['count'].rank(pct=True)
    
    # Create the figure
    fig = go.Figure()
    
    # Add text traces for each word
    for _, row in word_cloud_data.iterrows():
        # Get color from Viridis colorscale
        color_idx = min(int(row['color']*8), 7)  # Ensure index is within range
        
        fig.add_trace(go.Scatter(
            x=[row['x']],
            y=[row['y']],
            mode='text',
            text=[row['word']],
            textfont=dict(
                size=row['size'],
                color=px.colors.sequential.Viridis[color_idx]
            ),
            hoverinfo='text',
            hovertext=f"{row['word']}: {row['count']} occurrences",
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Word Cloud for {selected_user}' if selected_user != 'Overall' else 'Overall Word Cloud',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False
        ),
        hovermode='closest',
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_white'
    )
    
    # Print some debug info
    print(f"Generated word cloud with {len(word_cloud_data)} words after filtering stopwords and numbers.")
    
    return fig

def week_activity_map(selected_user: str, df: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced weekly activity heatmap showing hourly patterns.

    Args:
        selected_user: The user to analyze or 'Overall' for all users.
        df: The preprocessed DataFrame containing chat data.

    Returns:
        plotly Figure object representing the heatmap.
    """
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)

    if selected_user != 'Overall':
        df_copy = df_copy[df_copy['user'] == selected_user]
    
    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Extract day of week and hour
    df_copy['day_of_week'] = df_copy['date'].dt.day_name()
    df_copy['hour'] = df_copy['date'].dt.hour
    
    # Create pivot table for heatmap
    heatmap_data = df_copy.pivot_table(
        index='day_of_week', 
        columns='hour', 
        values='message',
        aggfunc='count',
        fill_value=0
    )
    
    # Reorder days and fill missing hours
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    all_hours = list(range(24))
    heatmap_data = heatmap_data.reindex(columns=all_hours, fill_value=0)
    
    # Create heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Message Count"),
        x=[str(h) for h in all_hours],
        y=day_order,
        color_continuous_scale='Viridis',
        title=f"Weekly Activity Pattern for {selected_user}" if selected_user != 'Overall' else "Overall Weekly Activity Pattern"
    )
    
    # Add hour labels
    hour_labels = [f"{h}:00" for h in all_hours]
    fig.update_xaxes(tickvals=list(all_hours), ticktext=hour_labels)
    
    # Add annotations for peak times if data exists
    if not heatmap_data.empty:
        peak_day_idx = heatmap_data.sum(axis=1).argmax()
        peak_hour = heatmap_data.sum(axis=0).argmax()
        peak_day = heatmap_data.index[peak_day_idx]
        
        # Find the coordinates of the absolute max value
        max_val = heatmap_data.max().max()
        if max_val > 0:
            peak_cell_coords = heatmap_data[heatmap_data == max_val].stack().idxmax()
            peak_cell_day, peak_cell_hour = peak_cell_coords
        
            fig.add_annotation(
                x=peak_cell_hour,
                y=peak_cell_day,
                text=f"Peak: {int(max_val)}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40,
                bgcolor="rgba(255, 255, 255, 0.7)"
            )

    # Update layout
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        coloraxis_colorbar=dict(
            title="Message Count",
        ),
        template='plotly_white'
    )
    
    return fig

def month_activity_map(selected_user: str, df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing message activity by month.

    Args:
        selected_user: The user to analyze or 'Overall' for all users.
        df: The preprocessed DataFrame containing chat data.

    Returns:
        plotly Figure object representing the bar chart.
    """
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)

    if selected_user != 'Overall':
        df_copy = df_copy[df_copy['user'] == selected_user]

    # Ensure month column exists
    if 'month' not in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
             df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['month'] = df_copy['date'].dt.month_name()
        df_copy['month_num'] = df_copy['date'].dt.month

    # Group by month and count messages
    busy_month = df_copy.groupby(['month', 'month_num']).size().reset_index(name='count')
    busy_month = busy_month.sort_values('month_num')


    fig = px.bar(busy_month, x='month', y='count', color='count',
                 color_continuous_scale='Viridis',
                 labels={'month': 'Month', 'count': 'Message Count'},
                 title=f"Monthly Activity for {selected_user}" if selected_user != 'Overall' else "Overall Monthly Activity")
    
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig

# Function for busiest hours analysis
def busiest_hours_analysis(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the count of messages per hour.

    Args:
        df: The preprocessed DataFrame containing chat data.

    Returns:
        pandas Series with hours as index and message count as values.
    """
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)
            
    # Ensure hour column exists
    if 'hour' not in df_copy.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
             df_copy['date'] = pd.to_datetime(df_copy['date'])
        df_copy['hour'] = df_copy['date'].dt.hour
        
    busiest_hours = df_copy['hour'].value_counts()
    return busiest_hours

def emoji_helper(selected_user: str, df: pd.DataFrame) -> go.Figure:
    """
    Analyzes emoji usage and creates a pie chart distribution for the top emojis.

    Note: This function is similar to `emoji_analysis` but might use 
    different filtering or presentation.

    Args:
        selected_user: The user to analyze or 'Overall' for all users.
        df: The preprocessed DataFrame containing chat data.

    Returns:
        plotly Figure object representing the emoji distribution pie chart.
    """
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)

    if selected_user != 'Overall':
        df_copy = df_copy[df_copy['user'] == selected_user]

    emojis = []
    for message in df_copy['message']:
        # Use emoji.EMOJI_DATA to check for emojis
        emojis.extend([c for c in message if emoji.is_emoji(c)])
        
    if not emojis:
        # Return an empty figure with a message if no emojis found
        fig = go.Figure()
        fig.add_annotation(
            text="No emojis found for the selected user.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="Emoji Distribution")
        return fig
        
    emoji_counts = Counter(emojis)
    emoji_df = pd.DataFrame(emoji_counts.most_common(), columns=['emoji', 'count'])

    # Ensure columns have correct names for px.pie if they differ from defaults
    emoji_df.columns = ['Emoji', 'Frequency'] 

    fig = px.pie(emoji_df.head(8), 
                 values='Frequency', 
                 names='Emoji',
                 title=f"Top 8 Emoji Distribution for {selected_user}" if selected_user != 'Overall' else "Overall Top 8 Emoji Distribution")

    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def analyze_and_plot_sentiment(selected_user: str, df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Performs enhanced sentiment analysis including distribution and time trends.

    Args:
        selected_user: The user to analyze or 'Overall' for all users.
        df: The preprocessed DataFrame containing chat data.

    Returns:
        Tuple[go.Figure, go.Figure]: A tuple containing two plotly figures:
            - dist_fig: Pie chart showing sentiment distribution.
            - trend_fig: Line chart showing sentiment trends over time.
    """
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)

    if selected_user != 'Overall':
        df_copy = df_copy[df_copy['user'] == selected_user]
    
    # Ensure we have date as datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Add month-year for grouping
    df_copy['month_year'] = df_copy['date'].dt.strftime('%Y-%m')
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each message
    sentiment_data = []
    for _, row in df_copy.iterrows():
        # Check if message is a string before analyzing
        if isinstance(row['message'], str):
            sentiment = sentiment_analyzer.polarity_scores(row['message'])
            sentiment_data.append({
                'user': row['user'],
                'date': row['date'],
                'month_year': row['month_year'],
                'message': row['message'],
                'positive': sentiment['pos'],
                'negative': sentiment['neg'],
                'neutral': sentiment['neu'],
                'compound': sentiment['compound'],
                'sentiment_category': 'Positive' if sentiment['compound'] > 0.05 else 
                                     'Negative' if sentiment['compound'] < -0.05 else 'Neutral'
            })
        else:
             # Handle non-string messages (e.g., NaN, if any) - assign neutral
             sentiment_data.append({
                'user': row['user'],
                'date': row['date'],
                'month_year': row['month_year'],
                'message': str(row['message']), # Ensure message is string
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'sentiment_category': 'Neutral'
            })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # --- Create Distribution Figure --- 
    if sentiment_df.empty:
        # Handle case with no sentiment data
        dist_fig = go.Figure().update_layout(title="No Sentiment Data Available")
        trend_fig = go.Figure().update_layout(title="No Sentiment Data Available")
        return dist_fig, trend_fig
        
    sentiment_counts = sentiment_df['sentiment_category'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Order categories
    category_order = ['Positive', 'Neutral', 'Negative']
    sentiment_counts['Sentiment'] = pd.Categorical(
        sentiment_counts['Sentiment'], 
        categories=category_order, 
        ordered=True
    )
    sentiment_counts = sentiment_counts.sort_values('Sentiment')
    
    # Calculate percentages
    total = sentiment_counts['Count'].sum()
    sentiment_counts['Percentage'] = (sentiment_counts['Count'] / total * 100).round(1)
    
    # Create distribution pie chart
    dist_fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        color='Sentiment',
        color_discrete_map={
            'Positive': '#4CAF50', # Green
            'Neutral': '#2196F3',  # Blue
            'Negative': '#F44336'   # Red
        },
        hole=0.4,
        title=f"Sentiment Distribution for {selected_user}" if selected_user != 'Overall' else "Overall Sentiment Distribution"
    )
    
    dist_fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>'
    )

    # --- Create Trend Figure --- 
    monthly_sentiment = sentiment_df.groupby('month_year').agg({
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean',
        'compound': 'mean',
        'message': 'count' # Keep track of message count per month
    }).reset_index()
    
    # Sort chronologically
    # Add a day to month_year for proper datetime conversion
    monthly_sentiment['date'] = pd.to_datetime(monthly_sentiment['month_year'] + '-01') 
    monthly_sentiment = monthly_sentiment.sort_values('date')
    # Format month_year for display
    monthly_sentiment['month_year_formatted'] = monthly_sentiment['date'].dt.strftime('%b %Y')
    
    trend_fig = go.Figure()
    
    # Add sentiment trend lines
    trend_fig.add_trace(go.Scatter(
        x=monthly_sentiment['month_year_formatted'],
        y=monthly_sentiment['positive'],
        mode='lines+markers',
        name='Positive',
        line=dict(color='#4CAF50', width=2)
    ))
    
    trend_fig.add_trace(go.Scatter(
        x=monthly_sentiment['month_year_formatted'],
        y=monthly_sentiment['negative'],
        mode='lines+markers',
        name='Negative',
        line=dict(color='#F44336', width=2)
    ))
    
    trend_fig.add_trace(go.Scatter(
        x=monthly_sentiment['month_year_formatted'],
        y=monthly_sentiment['compound'],
        mode='lines+markers',
        name='Overall Sentiment (Compound)',
        line=dict(color='#2196F3', width=3) 
    ))
    
    # Update trend layout
    trend_fig.update_layout(
        title={
            'text': f'Sentiment Trends for {selected_user}' if selected_user != 'Overall' else 'Overall Sentiment Trends',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Month',
        yaxis_title='Average Sentiment Score',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return dist_fig, trend_fig

def calculate_monthly_sentiment_trend(df: pd.DataFrame) -> go.Figure:
    """
    Calculates and plots the monthly trend of positive and negative sentiment.

    Args:
        df: The preprocessed DataFrame containing chat data.

    Returns:
        plotly Figure object showing the monthly sentiment trend.
    """
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)

    # Initialize the sentiment analyzer (ensure NLTK data is downloaded at module level)
    sid = SentimentIntensityAnalyzer()

    # Calculate sentiment for each message
    sentiment_scores = []
    valid_indices = []
    for index, message in df_copy['message'].items():
        if isinstance(message, str):
            sentiment_score = sid.polarity_scores(message)['compound']
            sentiment_scores.append(sentiment_score)
            valid_indices.append(index)
        else:
            sentiment_scores.append(0.0) # Assign neutral score for non-strings
            valid_indices.append(index)

    # Add sentiment scores to the copied DataFrame, handling potential length mismatch
    # Ensure sentiment_scores aligns with the original df_copy index
    sentiment_series = pd.Series(sentiment_scores, index=valid_indices)
    df_copy['sentiment_score'] = sentiment_series
    # Fill any potentially remaining NaNs if filtering occurred (shouldn't with current logic but safe)
    df_copy['sentiment_score'] = df_copy['sentiment_score'].fillna(0.0)

    # Convert 'date' column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Group data by month and calculate positivity and negativity percentages
    # Use .dt.to_period('M') for month grouping
    df_copy['month'] = df_copy['date'].dt.to_period('M')
    monthly_sentiment = df_copy.groupby('month').agg(
        positivity_percentage=('sentiment_score', lambda x: (x > 0.05).mean() * 100),
        negativity_percentage=('sentiment_score', lambda x: (x < -0.05).mean() * 100)
    ).reset_index() # Reset index to make 'month' a column

    # Convert Period index to string for plotting
    monthly_sentiment['month'] = monthly_sentiment['month'].astype(str)

    # Plot the trend
    fig = px.line(monthly_sentiment, x='month', y=['positivity_percentage', 'negativity_percentage'],
                  title='Monthly Sentiment Trend',
                  labels={'month': 'Month', 'value': 'Percentage of Messages', 'variable': 'Sentiment Type'},
                  color_discrete_map={'positivity_percentage': '#4CAF50', 'negativity_percentage': '#F44336'})
    
    # Ensure x-axis displays month labels correctly
    fig.update_xaxes(type='category')  
    fig.update_layout(hovermode='x unified', template='plotly_white')

    return fig

def message_count_aggregated_graph(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """
    Creates a pie chart of message counts per user and identifies the user with the most messages.

    Args:
        df: The preprocessed DataFrame containing chat data.

    Returns:
        Tuple[go.Figure, str]: A tuple containing:
            - fig: Plotly pie chart figure.
            - most_messages_winner: The username of the person who sent the most messages.
    """
    df_copy = df.copy()
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            pass
        else:
            df_copy = df_copy.reset_index(drop=True)
            
    # Filter out group notifications before counting
    user_df = df_copy[df_copy['user'] != 'group_notification']
    
    if user_df.empty:
         # Handle case with no user messages
        fig = go.Figure().update_layout(title="No User Messages Found")
        return fig, "No users found"

    subject_df = user_df.groupby('user')['message'].count().sort_values(ascending=False)
    
    if subject_df.empty:
        # Handle case where grouping resulted in empty series
        fig = go.Figure().update_layout(title="No User Messages Found After Grouping")
        return fig, "No users found"
        
    # Use idxmax() to find the index (user) with the maximum value
    most_messages_winner = subject_df.idxmax() 
    
    # Create a Pie chart using Plotly Express for simplicity
    fig = px.pie(subject_df, 
                 values=subject_df.values, 
                 names=subject_df.index,
                 title="Message Count Aggregated by User")
    
    fig.update_traces(textposition='inside', textinfo='percent+label')

    return fig, most_messages_winner

def conversation_starter_graph(df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """
    Creates a pie chart showing the distribution of conversation starters 
    and identifies the most frequent starter.

    Args:
        df: The preprocessed DataFrame containing chat data. 
            Requires a boolean column indicating conversation starts 
            (e.g., 'new_conversation') and a 'user' column.

    Returns:
        Tuple[go.Figure, str]: A tuple containing:
            - fig: Plotly pie chart figure.
            - most_frequent_starter: The username who started the most conversations.
    """
    df_copy = df.copy()
    
    # Handle potential date/index ambiguity before any operations
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # If 'date' is both in index and columns, reset index and drop old date index
            df_copy = df_copy.reset_index(drop=True)
        else:
             # If 'date' is only index, reset to make it a column
             df_copy = df_copy.reset_index() # Keep date column

    # Ensure 'new_conversation' column exists, calculate if missing
    # This calculation part is prone to the date error if not handled above
    if 'new_conversation' not in df_copy.columns:
        print("Warning: 'new_conversation' column not found. Calculating based on 60-min threshold.")
        # Ensure date is datetime for diff calculation
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
             df_copy['date'] = pd.to_datetime(df_copy['date'])
             
        df_copy = df_copy.sort_values('date') # Sort *after* resetting index if needed
        df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60
        conversation_threshold = 60
        df_copy['new_conversation'] = df_copy['time_diff'] > conversation_threshold
        # Handle the first message correctly
        if not df_copy.empty:
            df_copy.iloc[0, df_copy.columns.get_loc('new_conversation')] = True 
        
    # Filter out group notifications
    user_df = df_copy[df_copy['user'] != 'group_notification']

    # Filter for rows where a new conversation starts
    starters_df = user_df[user_df['new_conversation'] == True]

    # Group by user and count the number of times they started a conversation
    # Use .size() which is robust and doesn't rely on other columns existing
    subject_df = starters_df.groupby('user').size().sort_values(ascending=False)

    if subject_df.empty:
        # Handle case with no conversation starters found
        fig = go.Figure().update_layout(title="No Conversation Starters Found")
        return fig, "No starters found"

    # Create a Pie chart using Plotly Express for simplicity
    fig = px.pie(subject_df, 
                 values=subject_df.values, 
                 names=subject_df.index,
                 title="Conversation Starter Count by User")
                 
    fig.update_traces(textposition='inside', textinfo='percent+label')

    # Find the most frequent starter using idxmax()
    most_frequent_starter = subject_df.idxmax()
    
    return fig, most_frequent_starter

def conversation_size_aggregated_graph(df: pd.DataFrame) -> go.Figure:
    """
    Creates a line plot showing the average conversation size aggregated weekly over time.

    Args:
        df: The preprocessed DataFrame containing chat data. 
            Requires 'date' and 'message' columns. It will calculate 
            'conversation_id' if not present.

    Returns:
        go.Figure: Plotly line chart figure showing weekly average conversation size.
    """
    df_copy = df.copy()
    
    # Handle potential date/index ambiguity before any operations
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            df_copy = df_copy.reset_index(drop=True)
        else:
             df_copy = df_copy.reset_index()

    # Ensure 'conversation_id' exists, calculate if missing
    if 'conversation_id' not in df_copy.columns:
        print("Warning: 'conversation_id' column not found. Calculating based on 60-min threshold.")
        if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
             df_copy['date'] = pd.to_datetime(df_copy['date'])
             
        df_copy = df_copy.sort_values('date')
        df_copy['time_diff'] = df_copy['date'].diff().dt.total_seconds() / 60
        conversation_threshold = 60
        # Use .fillna(True) for the first row's comparison to handle NaT diff
        df_copy['new_conversation'] = (df_copy['time_diff'] > conversation_threshold).fillna(True)
        df_copy['conversation_id'] = df_copy['new_conversation'].cumsum()
        
    # Ensure date is datetime for aggregation
    elif not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
    # Group by conversation_id and aggregate size and mean date
    # Ensure we don't include group notifications in size calculation
    user_messages = df_copy[df_copy['user'] != 'group_notification']
    conversations_df = user_messages.groupby('conversation_id').agg(
        # Use 'message' or any guaranteed non-null column for size
        count=('message', 'size'), 
        # Use the first message date as the representative date for the conversation
        conv_date=('date', 'min') 
    ).reset_index()
    
    if conversations_df.empty:
        return go.Figure().update_layout(title="No Conversation Data Found")

    # Set index to the conversation date for resampling
    conversations_df.index = pd.to_datetime(conversations_df['conv_date'])
    
    # Resample weekly and calculate the mean count, fill missing weeks with 0
    # Use closed='left', label='left' to align weeks consistently
    weekly_avg_size = conversations_df['count'].resample('W', closed='left', label='left').mean().fillna(0)

    if weekly_avg_size.empty:
         return go.Figure().update_layout(title="No Weekly Data Found After Resampling")

    # Create a line plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_avg_size.index, y=weekly_avg_size.values, 
                           mode='lines', fill='tozeroy',
                           name='Avg Weekly Size'))
                           
    fig.update_layout(title="Weekly Average Conversation Size Over Time",
                      xaxis_title="Week Starting",
                      yaxis_title="Average Messages per Conversation",
                      template='plotly_white')
    return fig

def calculate_average_late_reply_time(df: pd.DataFrame, threshold_hours: int = 48) -> Tuple[go.Figure, pd.DataFrame, float]:
    """
    Calculate the average late reply time for users and display it with a graph.

    A reply is considered "late" if the time difference between consecutive messages 
    from different users exceeds the specified threshold.

    Args:
        df: DataFrame containing the chat data. Requires 'user' and 'date' columns.
        threshold_hours: Number of hours to consider a reply as "late" (default: 48 hours).

    Returns:
        Tuple[go.Figure, pd.DataFrame, float]: A tuple containing:
            - fig: Plotly bar chart showing average late reply times.
            - avg_late_reply_times_df: DataFrame with users and their average late reply times in hours.
            - overall_avg: The overall average late reply time in hours across all users.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Handle potential date/index ambiguity *before* sorting
    if 'date' in df_copy.index.names:
        if 'date' in df_copy.columns:
            # Ambiguous: Reset only the 'date' level from the index, discarding it
            df_copy = df_copy.reset_index(level='date', drop=True)
            # Reset any remaining index levels to columns if they exist
            if not isinstance(df_copy.index, pd.RangeIndex): 
                 df_copy = df_copy.reset_index()
        else:
             # Only in index: Reset all levels, making 'date' a column
             df_copy = df_copy.reset_index()

    # Filter out group notifications and media messages
    omitted_strings = ["image omitted", "media omitted", "video omitted", "sticker omitted", "audio omitted", "gif omitted"]
    df_copy = df_copy[(df_copy['user'] != 'group_notification') & 
                      (~df_copy['message'].str.lower().str.contains('|'.join(omitted_strings), na=False))]

    # Ensure date is sorted
    # Convert to datetime *just in case* before sorting
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values('date')

    # Calculate time difference and identify responses (similar to analyze_response_times)
    df_copy['time_diff_minutes'] = df_copy['date'].diff().dt.total_seconds() / 60
    df_copy['prev_user'] = df_copy['user'].shift(1)
    df_copy['is_response'] = (df_copy['user'] != df_copy['prev_user']) & (~df_copy['prev_user'].isna())
    
    # Filter for actual responses
    responses = df_copy[df_copy['is_response']].copy()

    # Filter for late replies (greater than threshold in minutes)
    threshold_minutes = threshold_hours * 60
    late_replies = responses[responses['time_diff_minutes'] > threshold_minutes]

    if late_replies.empty:
        # Handle case with no late replies
        fig = go.Figure().update_layout(title=f"No Late Replies Found (Threshold: {threshold_hours} hours)")
        empty_df = pd.DataFrame(columns=['user', 'Reply Time', 'Reply Time (Hours)'])
        return fig, empty_df, 0.0

    # Calculate average late reply time (in minutes) for each user who made a late reply
    # Group by the user who *made* the late reply (df_copy['user'])
    avg_late_reply_times = late_replies.groupby('user')['time_diff_minutes'].mean().reset_index()
    avg_late_reply_times.rename(columns={'time_diff_minutes': 'Reply Time'}, inplace=True)

    # Convert minutes to hours for better readability
    avg_late_reply_times['Reply Time (Hours)'] = avg_late_reply_times['Reply Time'] / 60
    
    # Sort by average reply time in descending order
    avg_late_reply_times_df = avg_late_reply_times.sort_values('Reply Time (Hours)', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        avg_late_reply_times_df, 
        x='user', 
        y='Reply Time (Hours)',
        title=f'Average Late Reply Time by User (Threshold: {threshold_hours} hours)',
        labels={'user': 'User', 'Reply Time (Hours)': 'Average Reply Time (Hours)'},
        color='Reply Time (Hours)',
        color_continuous_scale='Viridis'
    )
    
    # Add annotations for the values
    for i, row in avg_late_reply_times_df.iterrows():
        fig.add_annotation(
            x=row['user'],
            y=row['Reply Time (Hours)'],
            text=f"{row['Reply Time (Hours)']:.1f}h",
            showarrow=False,
            yshift=10
        )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title='Average Reply Time (Hours)',
        xaxis_title='User',
        height=600,
        # width=max(800, len(avg_late_reply_times_df) * 50), # Dynamic width
        template='plotly_white' 
    )
    
    # Calculate overall average late reply time in hours
    overall_avg = avg_late_reply_times_df['Reply Time (Hours)'].mean()
    
    # Add a horizontal line for the overall average
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=overall_avg,
        x1=len(avg_late_reply_times_df)-0.5,
        y1=overall_avg,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Add annotation for the overall average
    fig.add_annotation(
        xref="paper", x=0.5, # Position relative to plot area
        y=overall_avg,
        text=f"Overall Average: {overall_avg:.1f} hours",
        showarrow=False,
        yshift=10, # Adjust vertical position relative to the line
        font=dict(color="red"),
        bgcolor="rgba(255,255,255,0.7)" # Optional background for readability
    )
    
    return fig, avg_late_reply_times_df, overall_avg

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
    'analyze_reply_patterns',
    'analyze_message_types',
    'create_message_types_chart',
    'analyze_conversation_patterns',
    'create_conversation_patterns_chart',
    'analyze_user_interactions',
    'create_user_interaction_graph',
    'analyze_time_patterns',
    'analyze_message_length',
    'analyze_response_times',
    'analyze_topic_modeling',
    'analyze_emoji_usage',
    'analyze_sentiment_trends',
    'analyze_word_usage',
    'analyze_conversation_flow',
    'analyze_user_activity_patterns',
    'analyze_conversation_mood_shifts',
    'analyze_conversation_compatibility',
    'analyze_personality',
    'predict_future_activity',
    'create_plotly_wordcloud',
    'week_activity_map',
    'month_activity_map',
    'busiest_hours_analysis',
    'emoji_helper',
    'analyze_and_plot_sentiment',
    'calculate_monthly_sentiment_trend',
    'message_count_aggregated_graph',
    'conversation_starter_graph',       # Added
    'conversation_size_aggregated_graph', # Added
    'calculate_average_late_reply_time' # Added
] 