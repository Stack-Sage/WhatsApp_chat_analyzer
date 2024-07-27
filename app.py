import streamlit as st
import calendar
import pre_process
import pandas as pd
import helper
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from textblob import TextBlob


st.set_page_config(
    page_title="WhatsApp Chat Analyzer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",


)

# about_content = """
# <h1 style='color: black; text-shadow:4px 4px 4px white ;text-align: center;'>About The Website<br></h1>
#
# <p style = 'color:white;'>this is a streamlit app for analyzing whatsapp chat data. It provides insights and trends analysis based on the exported chat data. the app allows users to upload their whatsapp chat export files and explore various statistics and visualizations.</p>
#
# <p>For any questions or feedback:- <br> please contact us at <a href="mailto:official.adirajput@gmail.com">official.adirajput@gmail.com</a>.</p>
#
# <p>Connect with me on LinkedIn: <a href="https://www.linkedin.com/in/adarsh-pathania177/">adarsh-pathania177</a></p>
# """
#
# st.markdown(about_content, unsafe_allow_html=True)
# st.image('Black and Green Simple Business Youtube Thumbnail.png')
# Home Page
def home():
    st.header("")



# About Page
def about():
    about_content = """
    <h1 style='color: black; text-shadow:4px 4px 4px white ;text-align: center;'>About The Website<br></h1>

    <p style='color:white;'>This is a Streamlit app for analyzing WhatsApp chat data. It provides insights and trends analysis based on the exported chat data. The app allows users to upload their WhatsApp chat export files and explore various statistics and visualizations.</p>

    <p>For any questions or feedback:- <br> please contact us at <a href="mailto:official.adirajput@gmail.com">official.adirajput@gmail.com</a>.</p>

    <p>Connect with me on LinkedIn: <a href="https://www.linkedin.com/in/adarsh-pathania177/">adarsh-pathania177 </a></p>
    
    """

    st.markdown(about_content, unsafe_allow_html=True)
    st.image('Black and Green Simple Business Youtube Thumbnail.png')

    content = """
        ## WhatsApp Chat Analyzer

        Welcome to the WhatsApp Chat Analyzer! This Streamlit app empowers users to gain insights and trends from their exported chat data. Key features include:

        - **General Statistics:**
          - Total messages, words, media shared, and emojis used.
          - Distribution of messages throughout the day and the week.
          - Analysis of messages on each day and month.

        - **User Engagement:**
          - Identification of most engaged users.
          - Visualization of user engagement through bar graphs.

        - **Word Analysis:**
          - Word cloud visualization showcasing most used words.
          - Graphical representation of word usage frequency.

        - **Sentiment Analysis:**
          - Distribution of sentiment in the chat, categorized as very negative, negative, neutral, and positive.

        - **Hourly Distribution:**
          - Visualization of hourly distribution of messages.
          - Highlighting peak hours.

        - **Linear Regression:**
          - Linear regression analysis with a scatter plot of actual data and regression line.

        - **Active Users Heatmap:**
          - Heatmap displaying the most active hours of users.

        - **Individual User Analysis:**
          - Analysis of individual users, including word cloud and graphical representation.

        Feel free to upload your WhatsApp chat export file and explore the rich set of statistical and visualizations features! For any questions or feedback, please contact us at [official.adirajput@gmail.com](mailto:official.adirajput@gmail.com).
        Go to the Home Section for analysis
        """


    st.markdown(content, unsafe_allow_html=True)

st.sidebar.title("📱 Whatsapp Chat Analyzer")
st.sidebar.image("px4.jpg",width=200)

# Create a sidebar for navigation
page = st.sidebar.radio("Navigation", ["About", "Home"], index =0)

# Display the selected page
if page == "Home":
    home()
elif page == "About":
    about()




st.sidebar.markdown('   Hey Fellas!👋...Start Here 👇🏻...')


st.sidebar.markdown("Please upload your exported WhatsApp chat...(without media & in 24-hour format)")

uploaded_file = st.sidebar.file_uploader("Choose a file 🔍")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df, message_by_hour, most_active_user = pre_process.preprocess(data)

    # fetch unique users
    user_list = df["user"].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis", user_list)

    if st.sidebar.button("Show Analysis 📊"):


        num_messages, words, num_media_messages, total_emojis = helper.fetch_stats(selected_user, df)

        st.markdown("<h1 style='color:#a6f476  ; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: center;'>📊 Insights and Trends Analysis 📈</h1> <br>",unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown("<h3 style='color:#a6f476 ;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: left;'>Total Messages 🗨️</h3>",
                        unsafe_allow_html=True)

            st.title(num_messages)

        with col2:

            st.markdown("<h3 style='color:#a6f476;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: left;'>Total words 📝</h3>",
                        unsafe_allow_html=True)
            st.title(words)

        with col3:

            st.markdown("<h3 style='color:#a6f476 ;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: left;'>Media Shared 📹</h3>",
                        unsafe_allow_html=True)
            st.title(num_media_messages)

        with col4:

            st.markdown("<h3 style='color:#a6f476;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: left;'>Emoji Shared 😀</h3>",
                        unsafe_allow_html=True)
            st.title(total_emojis)

        # Most Busy Users

        x, new_df = helper.most_busy_users(df)

        num_data_points = len(x.index)
        figsize = (10,10) if num_data_points >= 5 else (2, 1)
        bar_width = 0.4 if num_data_points >= 4 else 0.01  # Adjust the bar width as needed
        bar_height = 40 if num_data_points >= 5 else 15  # Adjust the bar height as needed

        # Create a black background figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('white')  # Set the background color of the entire figure to black
        ax.set_facecolor('white')

        cl1, cl2 = st.columns(2)

        with cl1:  # code for displaying stats
            st.markdown("<br> <h2 style='color:#ade276; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align:left;'>Most Engaged Users 🌟</h2>", unsafe_allow_html=True)
            st.dataframe(new_df, width=450)

        with cl2:
            # Check if the number of users is greater than 5
            if len(x)>4:
                st.markdown("<br><h2 style='color: #ade276; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: center;'>📊 Graphical Representation 📈</h2>", unsafe_allow_html=True)



                # Draw the background gradient image
                direction = 1
                extent = (0, 1, 0, 1)
                transform = ax.transAxes
                cmap = plt.cm.plasma
                cmap_range = (0.2, 0.8)
                alpha = 0.5
                phi = direction * np.pi / 2
                v = np.array([np.cos(phi), np.sin(phi)])
                X = np.array([[v @ [1, 0], v @ [1, 1]],
                              [v @ [0, 0], v @ [0, 1]]])
                a, b = cmap_range
                X = a + (b - a) / X.max() * X
                im = ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent,
                               transform=transform, alpha=alpha)

                # Set the color of the bars using a single color
                bar_color = 'pink'
                ax.barh(x.index, x.values, color=bar_color, edgecolor='black', linewidth=1.5)

                # Set the color of y-axis and x-axis labels to a contrasting color
                plt.yticks(color='white', size='18')
                plt.ylabel("Users", color='red', size='18')
                plt.xticks(color='white', size='18')
                plt.xlabel("No of Messages", color='red', size='18')

                # Set the color of axes lines to a contrasting color
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')

                # Set the background color of the entire figure to black
                fig.patch.set_facecolor('black')

                st.pyplot(fig)

            else:
                st.markdown("<br><h2 style='color:#ade276;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align: center;'>📊 Graphical Representation 📈</h2>",
                            unsafe_allow_html=True)

                # Create a figure
                fig, ax = plt.subplots(figsize=(5,5))

                # Set the color of the background to black
                fig.patch.set_facecolor('black')

                # Set the color of the pie chart
                colors = ['blue','green', 'pink', 'lightcoral']

                labels = [f"{user}\n{messages} messages" for user, messages in x.items()]
                ax.pie(x.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                wedges, texts, autotexts = ax.pie(x.values, labels=x.keys(), autopct='%1.1f%%', startangle=90,
                                                  colors=colors)

                # Set the color of the labels to white
                for text, autotext in zip(texts, autotexts):
                    text.set_color('white')
                    autotext.set_color('white')

                st.pyplot(fig)

        sa1,sa2 = st.columns(2)
        #word map
        with sa1:
            df_wc = helper.create_wordcloud(selected_user, df)
            word_freq = pd.Series(df_wc.words_).sort_values(ascending=False)
            word_counts = df['messages'].str.split().explode().value_counts().reset_index()
            word_counts.columns = ['Word', 'Count']
            merged_df = word_freq.reset_index().rename(columns={'index': 'Word', 0: 'Frequency'})
            merged_df = merged_df.merge(word_counts, on='Word', how='left').fillna({'Count': 1})
            merged_df = merged_df.sort_values(by=['Frequency'], ascending=False).head(20)
            merged_df.columns = ['Word', 'Usage Frequency', 'Usage Count']

            # Keep only the desired columns for display
            selected_columns = ['Word', 'Usage Count']
            st.markdown(
                "<br><h2 style='color:#ade276;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5);text-align:left ;'>Most Used words </h2>",
                unsafe_allow_html=True)
            st.dataframe(merged_df[selected_columns], width=400)

        with sa2:
                st.markdown(
                    "<br><h2 style='color:#ade276 ;text-shadow: 0 0 2px rgba(255, 255, 255, 0.5);text-align: center;'>📊 Graphical Representation 📈</h2>",
                    unsafe_allow_html=True)
                # Create a figure and axis
                fig, ax = plt.subplots(figsize=(11, 11))

                # Draw the background gradient image
                direction = 1
                extent = (0, 1, 0, 1)
                transform = ax.transAxes
                cmap = plt.cm.viridis
                cmap_range = (0.2, 0.8)
                alpha = 0.5
                phi = direction * np.pi / 2
                v = np.array([np.cos(phi), np.sin(phi)])
                X = np.array([[v @ [1, 0], v @ [1, 1]],
                              [v @ [0, 0], v @ [0, 1]]])
                a, b = cmap_range
                X = a + (b - a) / X.max() * X
                im = ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent,
                               transform=transform, alpha=alpha)

                # Set the color of the bars using a single color
                bar_color = 'skyblue'
                ax.barh(merged_df['Word'], merged_df['Usage Count'], color=bar_color, edgecolor='black', linewidth=1.5)

                # Set the color of y-axis and x-axis labels to a contrasting color
                plt.yticks(color='white', size='18')
                plt.ylabel("Words", color='yellow', size='18')
                plt.xticks(color='white', size='18')
                plt.xlabel("Usage Count", color='yellow', size='18')

                # Set the color of axes lines to a contrasting color
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['left'].set_color('white')

                # Set the background color of the entire figure to black
                fig.patch.set_facecolor('black')

                # Display the plot in Streamlit
                st.pyplot(fig)

        #individual analysis of every user ------heatmap
        st.markdown(
            "<br><h2 style='color:#347af6; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5);text-align:center;'>Most active users</h2>",
            unsafe_allow_html=True)

        # heatmap of most avtive users
        heatmap_data = df.pivot_table(index='user', columns='hour', aggfunc='size', fill_value=0)

        # Identify the most active hour for each user
        most_active_hour_per_user = df.groupby(['user', 'hour']).size().reset_index(name='message_count')
        idx = most_active_hour_per_user.groupby(['user'])['message_count'].transform(max) == \
              most_active_hour_per_user['message_count']
        most_active_hour_per_user = most_active_hour_per_user[idx]

        # Create a dictionary mapping users to their most active hours
        user_active_hours = dict(zip(most_active_hour_per_user['user'], most_active_hour_per_user['hour']))

        # Create a new column in the DataFrame indicating the most active hour for each user
        df['most_active_hour'] = df['user'].map(user_active_hours)
        # Customize color of the axis labels area

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(8, 4))

        # Customize color of the axis labels area
        ax.tick_params(axis='both', colors='black')

        # Draw the background gradient image
        direction = 1
        extent = (0, 1, 0, 1)
        transform = ax.transAxes
        cmap = plt.cm.Blues
        cmap_range = (0.2, 0.8)
        alpha = 0.5
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        im = ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent,
                       transform=transform, alpha=alpha)

        # Customize the heatmap
        sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='g', cbar_kws={'label': 'Number of Messages'},
                    linewidths=.5, linecolor='black', ax=ax, xticklabels=2)

        # Customize other plot elements
        plt.title('User Most Active Hours of Users')
        plt.xlabel('Hour of the Day', size=10)
        plt.xticks(size=10)
        plt.yticks(size=10)
        plt.ylabel('User', size=10)

        # Set the facecolor of the entire figure to the gradient background
        fig.patch.set_facecolor(im.cmap(im.norm(0.5)))

        # Display the heatmap in Streamlit
        st.pyplot(fig)

        # Overall hourly analysis
        # Plotting the distribution of hourly messages


        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10,5))

        # Draw the background gradient
        direction = 1
        extent = (0, 1, 0, 1)
        transform = ax.transAxes
        cmap = plt.cm.magma
        cmap_range = (0.2, 0.8)
        alpha = 0.5
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        im = ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent,
                       transform=transform, alpha=alpha)



        # Plot the line chart
        sns.lineplot(x='hour', y='message_count', data=message_by_hour, marker='o', ax=ax, color='white')

        st.markdown(
            "<br><h2 style='color:#e56d9b; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align:center;'>Hourly Distribution</h2>",
            unsafe_allow_html=True)

        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Messages')
        plt.grid(True)

        # Set x-axis ticks with a difference of 2
        plt.xticks(range(0, 24, 2))

        # Highlight peak hours (customize the threshold for highlighting)
        peak_threshold = message_by_hour['message_count'].max() * 0.8
        plt.axhline(peak_threshold, color='red', linestyle='--', label='Peak Hours Threshold')


        # Customize other plot elements
        plt.legend()
        ax.tick_params(axis='both', colors='white', size=14)

        plt.xlabel('Hour of the Day', color='white', size=14)
        plt.ylabel('Number of Messages', color='white', size=14)
        fig.patch.set_facecolor('black')

        # Display the plot in Streamlit
        st.pyplot(fig)


        # sentiment analysis of the char ----------------------------sentiment analysis

        st.markdown(
            "<br><h2 style='color:#006bff ; text-shadow: 0 0 2px rgba(255, 255, 255, 0.8); text-align:center;'>Sentiment Analysis</h2>",
            unsafe_allow_html=True)


        df['sentiment'] = df['messages'].apply(lambda x: TextBlob(x).sentiment.polarity)

        # Map sentiment polarity to custom labels
        df['sentiment_label'] = pd.cut(df['sentiment'], bins=[-1, -0.5, 0, 0.5, 1, 1.5],
                                       labels=['very negative', 'negative', 'neutral', 'positive','very positive'])
        # Display the distribution of sentiments
        st.write("Sentiment Distribution in Dataset:")
        sentiment_chart = st.bar_chart(df['sentiment_label'].value_counts())





        # # Create a bar plot
        # fig, ax = plt.subplots(figsize=(8,4))
        # fig.patch.set_facecolor('black')

        # # Add a gradient background
        # direction = 1
        # extent = (0, 1, 0, 1)
        # transform = ax.transAxes
        # cmap = plt.cm.rainbow
        # cmap_range = (0.2, 0.8)
        # alpha = 0.5
        # phi = direction * np.pi / 2
        # v = np.array([np.cos(phi), np.sin(phi)])
        # X = np.array([[v @ [1, 0], v @ [1, 1]],
        #               [v @ [0, 0], v @ [0, 1]]])
        # a, b = cmap_range
        # X = a + (b - a) / X.max() * X
        # im = ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent,
        #                transform=transform, alpha=alpha, zorder=-1)
        #
        # # Create the count plot
        # ax = sns.countplot(x='sentiment_label', data=df, order=df['sentiment_label'].value_counts().index,
        #                    palette='viridis', edgecolor='black', saturation=0.7,linewidth =1,width=0.3)
        #
        # # Add text annotations for each bar
        # for p in ax.patches:
        #     ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 1., p.get_height()),
        #                 ha='center', va='center', fontsize=10, color='black', xytext=(0,5),
        #                 textcoords='offset points')
        #
        #
        # # Customize other plot elements
        # plt.title('Sentiment Distribution in Chat')
        # plt.xticks(color='white')
        # plt.tick_params(color='white')
        # plt.yticks(color='white')
        # plt.xlabel('Sentiment',color='white')
        # plt.ylabel('Count',color = 'white')
        #
        # # Display the plot in Streamlit
        # st.pyplot(fig)

        st.markdown(
            "<br><h2 style='color: #a6ee52; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align:center;'>Linear Regression </h2>",
            unsafe_allow_html=True)

        # Perform linear regression
        model, mae = helper.perform_linear_regression(df)
        rmse, r_squared = helper.calculate_regression_metrics(model, df)

        # Scatterplot to visualize data and predictions
        plt.figure(figsize=(8, 4))

        # Draw the background gradient image
        direction = 1
        extent = (0, 1, 0, 1)
        transform = plt.gca().transAxes
        cmap = plt.cm.viridis
        cmap_range = (0.2, 0.8)
        alpha = 0.5
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        plt.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent,
                   transform=transform, alpha=alpha)

        # Plot actual data
        sns.scatterplot(x=df['hour'], y=df['message_count'], label='Actual Data', linewidth=1)

        # Predict values using the model
        y_pred = model.predict(df[['hour']])

        # Plot the regression line
        sns.lineplot(x=df['hour'], y=y_pred, label='Regression Line', color='red', linewidth=1)

        plt.xlabel('Hour of the Day',color='white')
        plt.ylabel('Number of Messages',color='white')
        plt.tick_params(color='white')

        plt.xticks(range(0,24,2),color='white')
        plt.yticks(color='white')

        plt.legend()

        # Set the facecolor of the entire figure to black
        plt.gcf().patch.set_facecolor('black')


        # Display the plot in Streamlit
        st.pyplot(plt)

        # weekly analysis  -----------------------------------------------weekly
        st.markdown(
            "<br><h2 style='color:#006bff; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align:center;'>No of Message on each day of the week</h2>",
            unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8,4))

        # Add gradient background to the entire figure
        direction = 1
        extent = (0, 1, 0, 1)
        transform = ax.transAxes
        cmap = plt.cm.turbo
        cmap_range = (0.2, 0.8)
        alpha = 0.5
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent, alpha=alpha,transform=transform,
                  zorder=-1)


        sns.countplot(x='day_name', data=df, order=calendar.day_name, palette='viridis', ax=ax,edgecolor ='black',linewidth=1,width=0.45)

        plt.xlabel('Day of the Week',color='white')
        plt.ylabel('Number of Messages',color ='white')



        plt.tick_params(color='white')
        plt.xticks(color ='white')
        plt.yticks(color='white')



        fig.patch.set_facecolor('black')
        # Display the plot in Streamlit
        st.pyplot(fig)

        # Daily month  analysis ----------------------------------------daily month
        st.markdown(
            "<br><h2 style='color:#5a5ae3 ; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5); text-align:center;'>No of Messages on Each Day of the Month</h2>",
            unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Add gradient background to the entire figure
        direction = 1
        extent = (0, 1, 0, 1)
        transform = ax.transAxes
        cmap = plt.cm.rainbow
        cmap_range = (0.2, 0.8)
        alpha = 0.5
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent, alpha=alpha,
                  transform=transform,
                  zorder=-1)

        # Countplot
        sns.countplot(x='day', data=df, palette='winter', ax=ax,linewidth=1,edgecolor ='black')

        plt.xlabel('Day of the Month', color='white')
        plt.ylabel('Number of Messages', color='white')

        plt.tick_params(color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')

        fig.patch.set_facecolor('black')
        # Display the plot in Streamlit
        st.pyplot(fig)

        # monthly analysis =----------------------------------------------monthly
        st.markdown(
            "<br><h2 style='color: #70efe3 ; text-shadow: 0 0 2px rgba(255, 255, 255, 0.5);  text-align:center;'>No of Messages on Each Month</h2>",
            unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10,5))

        # Add gradient background to the entire figure
        direction = 1
        extent = (0, 1, 0, 1)
        transform = ax.transAxes
        cmap = plt.cm.winter
        cmap_range = (0.2, 0.8)
        alpha = 0.5
        phi = direction * np.pi / 2
        v = np.array([np.cos(phi), np.sin(phi)])
        X = np.array([[v @ [1, 0], v @ [1, 1]],
                      [v @ [0, 0], v @ [0, 1]]])
        a, b = cmap_range
        X = a + (b - a) / X.max() * X
        ax.imshow(X, interpolation='bicubic', clim=(0, 1), aspect='auto', cmap=cmap, extent=extent, alpha=alpha,
                  transform=transform,
                  zorder=-1)

        # Countplot
        sns.countplot(x='month', data=df, order=calendar.month_name[1:], palette='plasma', ax=ax,edgecolor='black',width=0.6,linewidth =1)

        plt.xlabel('Month', color='white')
        plt.ylabel('Number of Messages', color='white')

        plt.tick_params(color='white')
        plt.xticks(color='white',fontsize=8)
        plt.yticks(color='white')

        fig.patch.set_facecolor('black')
        # Display the plot in Streamlit
        st.pyplot(fig)



        content = """
        <br><h1 style='color: black; text-shadow: 0 0 5px rgba(255, 255, 255, 0.8), 0 0 10px rgba(255, 255, 255, 0.5), 0 0 20px rgba(255, 255, 255, 0.3);text-align: center;'>About The Website</h1><br>
        
        <p style = 'text-shadow:1px 2px 1px blue>This is a Streamlit app for analyzing WhatsApp chat data. It provides insights and trends analysis based on the exported chat data. The app allows users to upload their WhatsApp chat export files and explore various statistics and visualizations.</p>
        
        <p>For any questions or feedback:- <br> Please contact us at <a href="mailto:official.adirajput@gmail.com">official.adirajput@gmail.com</a>.</p>
        
        <p>Connect with me on LinkedIn: <a href="https://www.linkedin.com/in/adarsh-pathania177/">adarsh-pathania177</a></p>
        """

        st.markdown(content, unsafe_allow_html=True)

        about_content = """
            ## WhatsApp Chat Analyzer

            Welcome to the WhatsApp Chat Analyzer! This Streamlit app empowers users to gain insights and trends from their exported chat data. Key features include:

            - **General Statistics:**
              - Total messages, words, media shared, and emojis used.
              - Distribution of messages throughout the day and the week.
              - Analysis of messages on each day and month.

            - **User Engagement:**
              - Identification of most engaged users.
              - Visualization of user engagement through bar graphs.

            - **Word Analysis:**
              - Word cloud visualization showcasing most used words.
              - Graphical representation of word usage frequency.

            - **Sentiment Analysis:**
              - Distribution of sentiment in the chat, categorized as very negative, negative, neutral, and positive.

            - **Hourly Distribution:**
              - Visualization of hourly distribution of messages.
              - Highlighting peak hours.

            - **Linear Regression:**
              - Linear regression analysis with a scatter plot of actual data and regression line.

            - **Active Users Heatmap:**
              - Heatmap displaying the most active hours of users.

            - **Individual User Analysis:**
              - Analysis of individual users, including word cloud and graphical representation.

            Feel free to upload your WhatsApp chat export file and explore the rich set of statistical and visualizations features! For any questions or feedback, please contact us at [official.adirajput@gmail.com](mailto:official.adirajput@gmail.com).
            """

        st.markdown(about_content, unsafe_allow_html=True)


