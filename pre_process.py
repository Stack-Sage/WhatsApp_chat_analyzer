import re
import pandas as pd
import calendar



def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_messages': messages, 'message_date': dates})
    df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_messages']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['messages'] = messages
    # Convert 'messages' column to strings
    df['messages'] = df['messages'].astype(str)

    df.drop(columns=['user_messages'], inplace=True)
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['only_date'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['day_of_week'].apply(lambda x: calendar.day_name[x])

    # Create a new DataFrame to count messages per hour
    message_by_hour = df.groupby(df['hour']).size().reset_index(name='message_count')

    # Merge the new DataFrame with your original DataFrame
    df = df.merge(message_by_hour, left_on='hour', right_on='hour', how='left')

    # Filter the DataFrame to exclude the row with the specified name
    df = df[df['user'] != 'You joined a group via invite in the community']
    df = df.dropna(subset=['messages'])

    # Reset the index of the filtered DataFrame
    df.reset_index(drop=True, inplace=True)

    if 'group_notification' in df['user'].values:
        df = df[df['user'] != 'group_notification']

    most_active_hour_per_user = df.groupby(['user', 'hour']).size().reset_index(name='message_count')
    idx = most_active_hour_per_user.groupby(['user'])['message_count'].transform(max) == most_active_hour_per_user[
        'message_count']
    most_active_hour_per_user = most_active_hour_per_user[idx]

    # Create a separate DataFrame for most active hours per user
    most_active_hour = most_active_hour_per_user[['user', 'hour']]



    # Merge the most active hour DataFrame with your original DataFrame
    df = df.merge(most_active_hour, on='user', how='left', suffixes=('', '_most_active_hour'))

    return df,message_by_hour,most_active_hour

