import re
import pandas as pd
import calendar

def preprocess(data):
    if not isinstance(data, str):
        raise ValueError("Input data must be a string.")

    # Remove thin spaces (U+202F) before AM/PM if present
    data = data.replace("\u202F", " ")

    # Updated regex to match both 24-hour and AM/PM formats correctly
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?:\s?(?:AM|PM|am|pm))?\s-\s'

    messages = re.split(pattern, data)[1:]  # Extract messages
    dates = re.findall(pattern, data)  # Extract timestamps

    if not messages or not dates:
        raise ValueError("No messages found. Check the chat format.")

    # Create DataFrame
    df = pd.DataFrame({'user_messages': messages, 'message_date': dates})

    # Convert to datetime, ensuring correct parsing
    df['message_date'] = df['message_date'].str.strip(' -')  # Remove trailing ' -'
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p', errors='coerce')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extract Usernames & Messages
    users, messages = [], []
    for message in df['user_messages']:
        entry = re.split(r'([\w\W]+?):\s', message, maxsplit=1)
        if len(entry) >= 3:
            users.append(entry[1])  # Extract username
            messages.append(entry[2])  # Extract message
        else:
            users.append('group_notification')  # System messages
            messages.append(entry[0] if entry else "")

    df['user'] = users
    df['messages'] = messages
    df['messages'].fillna("", inplace=True)

    df.drop(columns=['user_messages'], inplace=True)

    # Extract Date & Time Features
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['only_date'] = df['date'].dt.date
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['day_of_week'].apply(lambda x: calendar.day_name[x])

    # Messages Per Hour
    message_by_hour = df.groupby('hour').size().reset_index(name='message_count')

    df = df.merge(message_by_hour, on='hour', how='left')

    # Remove system messages
    df = df[~df['user'].str.contains('joined a group', na=False)]
    df = df[df['user'] != 'group_notification']

    # Most Active Hour Per User
    most_active_hour_per_user = df.groupby(['user', 'hour']).size().reset_index(name='message_count')
    idx = most_active_hour_per_user.groupby(['user'])['message_count'].transform('max') == most_active_hour_per_user['message_count']
    most_active_hour_per_user = most_active_hour_per_user[idx]

    most_active_hour = most_active_hour_per_user[['user', 'hour']]
    df = df.merge(most_active_hour, on='user', how='left', suffixes=('', '_most_active_hour'))

    return df, message_by_hour, most_active_hour
