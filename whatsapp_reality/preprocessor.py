"""
Preprocessor module for WhatsApp chat data.
"""

import re
import pandas as pd
import numpy as np
from dateutil.parser import parse
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple, List

def preprocess(data: str) -> pd.DataFrame:
    """
    Preprocess WhatsApp chat data from a text export.
    
    Args:
        data: Raw text data from WhatsApp chat export
        
    Returns:
        pd.DataFrame: Processed DataFrame with chat data
        
    Example:
        >>> with open('chat.txt', 'r', encoding='utf-8') as file:
        ...     chat_data = file.read()
        >>> df = preprocess(chat_data)
    """
    # Define patterns for different formats
    android_12hr_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s*-\s*'
    android_24hr_pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s*-\s*'
    ios_12hr_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\s(?:AM|PM)\]'
    ios_24hr_pattern = r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\]'
    
    # Try to detect the format
    if re.search(android_24hr_pattern, data):
        pattern = android_24hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: x.strip('- '))
    
    elif re.search(android_12hr_pattern, data):
        pattern = android_12hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: x.strip('- '))
    
    elif re.search(ios_12hr_pattern, data):
        pattern = ios_12hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: re.sub(r'[\[\]]', '', x))
    
    elif re.search(ios_24hr_pattern, data):
        pattern = ios_24hr_pattern
        messages = re.split(pattern, data)[1:]
        dates = re.findall(pattern, data)
        df = pd.DataFrame({'user_message': messages, 'message_date': dates})
        df['message_date'] = df['message_date'].apply(lambda x: re.sub(r'[\[\]]', '', x))
    
    else:
        raise ValueError("Unsupported WhatsApp chat format. Please ensure your chat export is from WhatsApp and follows either Android or iOS format.")

    # Parse dates
    df['date'] = df['message_date'].apply(lambda x: parse(x, fuzzy=True))
    df.index = df['date']

    # Extract users and messages
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name exists
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df['Message Length'] = df['message'].apply(lambda x: len(x.split()))

    # Add conversation analysis
    conv_codes, conv_changes = _cluster_into_conversations(df)
    df['Conv code'] = conv_codes
    df['Conv change'] = conv_changes

    # Add reply analysis
    is_reply, sender_changes = _find_replies(df)
    df['Is reply'] = is_reply
    df['Sender change'] = sender_changes

    # Add user-specific columns
    for subject in df['user'].unique():
        df[subject] = df['user'].apply(lambda x: 1 if x == subject else 0)
        df[f"{subject}_mlength"] = df[subject].values * df['Message Length']

    # Add time-based columns
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Add time periods
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))
    df['period'] = period

    # Calculate reply times
    df = _add_reply_times(df)

    # Drop intermediate columns
    df.drop(columns=['user_message', 'message_date'], inplace=True)

    return df

def _cluster_into_conversations(df: pd.DataFrame, threshold_mins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to cluster messages into conversations."""
    threshold_time = np.timedelta64(threshold_mins, 'm')
    conv_delta = df.index.values - np.roll(df.index.values, 1)
    conv_delta[0] = np.timedelta64(0)
    
    conv_changes = conv_delta > threshold_time
    conv_changes_indices = np.where(conv_changes)[0]
    
    if len(conv_changes_indices) == 0:
        return np.zeros(len(df)), np.zeros(len(df), dtype=bool)
        
    conv_codes = np.zeros(len(df))
    current_code = 0
    last_change = 0
    
    for change_idx in conv_changes_indices:
        conv_codes[last_change:change_idx] = current_code
        current_code += 1
        last_change = change_idx
    
    conv_codes[last_change:] = current_code
    return conv_codes, conv_changes

def _find_replies(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to identify replies in conversations."""
    user_encoder = OrdinalEncoder()
    message_senders = user_encoder.fit_transform(df['user'].values.reshape(-1, 1))
    
    sender_changed = (np.roll(message_senders, 1) - message_senders).reshape(-1) != 0
    sender_changed[0] = False
    
    is_reply = sender_changed & ~df['Conv change']
    return is_reply, sender_changed

def _add_reply_times(df: pd.DataFrame) -> pd.DataFrame:
    """Helper function to calculate reply times."""
    reply_times = []
    for i in range(len(df)):
        if df['Is reply'].iloc[i]:
            reply_time = (df.index[i] - df.index[i-1]).total_seconds() / 60  # Convert to minutes
            reply_times.append(reply_time)
        else:
            reply_times.append(0)
    
    df['Reply Time'] = reply_times
    return df

__all__ = ['preprocess'] 