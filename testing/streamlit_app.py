import streamlit as st
import pandas as pd
from whatsapp_reality import preprocess, analyzer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")

st.title("WhatsApp Chat Analyzer")

st.markdown("""
Upload your WhatsApp chat export file (`.txt`) to analyze the conversation.
""")

uploaded_file = st.file_uploader("Choose a WhatsApp chat export file (.txt)", type="txt")

if uploaded_file is not None:
    try:
        # Read chat data
        chat_data = uploaded_file.getvalue().decode("utf-8")
        
        # Preprocess data
        with st.spinner('Preprocessing chat data...'):
            df = preprocess(chat_data)
        st.success("Preprocessing complete!")

        st.header("Analysis Results")

        # --- Analysis Sections ---

        # 1. Basic Statistics
        with st.expander("1. Basic Statistics", expanded=False):
            st.subheader("Overall Chat Statistics")
            messages, words, media, links = analyzer.fetch_stats('Overall', df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Messages", messages)
            col2.metric("Total Words", words)
            col3.metric("Media Messages", media)
            col4.metric("Links Shared", links)
            
            st.subheader("User-Specific Statistics")
            user_stats_list = []
            for user in df['user'].unique():
                if user != 'group_notification':
                    m, w, md, l = analyzer.fetch_stats(user, df)
                    user_stats_list.append({'User': user, 'Messages': m, 'Words': w, 'Media': md, 'Links': l})
            user_stats_df = pd.DataFrame(user_stats_list)
            st.dataframe(user_stats_df)


        # 2. User Activity Analysis
        with st.expander("2. User Activity Analysis (Most Busy Users)", expanded=False):
            fig_busy_users, df_percent = analyzer.most_busy_users(df)
            st.subheader("Most Active Users (% of total messages)")
            st.dataframe(df_percent)
            st.plotly_chart(fig_busy_users, use_container_width=True)

        # 3. Plotly Word Cloud Generation
        # with st.expander("3. Word Cloud (Plotly)", expanded=False):
        #     st.subheader("Most Frequent Words (Interactive)")
        #     with st.spinner("Generating Plotly word cloud..."):
        #         plotly_wordcloud_fig = analyzer.create_plotly_wordcloud('Overall', df, True)
        #     st.plotly_chart(plotly_wordcloud_fig, use_container_width=True)

        # 4. Matplotlib Word Cloud Generation
        with st.expander("4. Word Cloud (Static)", expanded=False):
            st.subheader("Most Frequent Words (Image)")
            with st.spinner("Generating Matplotlib word cloud..."):
                wordcloud_obj = analyzer.create_wordcloud('Overall', df)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud_obj)
                ax.axis('off')
                st.pyplot(fig)

        # 5. Most Common Words
        with st.expander("5. Most Common Words (Bar Chart)", expanded=False):
            st.subheader("Top 20 Most Frequent Words")
            with st.spinner("Analyzing common words..."):
                common_words_fig = analyzer.most_common_words('Overall', df)
            st.plotly_chart(common_words_fig, use_container_width=True)

        # 6. Emoji Analysis (Original Pie Chart)
        with st.expander("6. Emoji Analysis (Top Emojis)", expanded=False):
            st.subheader("Most Used Emojis")
            with st.spinner("Analyzing emojis..."):
                try:
                    emoji_fig = analyzer.emoji_analysis('Overall', df)
                    st.pyplot(emoji_fig)
                except Exception as e:
                    st.warning(f"Could not generate emoji pie chart: {e}")

        # 7. Emoji Helper (Similar Pie Chart)
        with st.expander("7. Emoji Helper (Top Emojis - Alternative)", expanded=False):
             st.subheader("Most Used Emojis (Plotly)")
             with st.spinner("Analyzing emojis (helper)..."):
                 try:
                     emoji_helper_fig = analyzer.emoji_helper('Overall', df)
                     st.plotly_chart(emoji_helper_fig, use_container_width=True)
                 except Exception as e:
                     st.warning(f"Could not generate emoji helper chart: {e}")


        # 8. Timeline Analysis
        with st.expander("8. Timeline Analysis", expanded=False):
            st.subheader("Monthly Message Timeline")
            with st.spinner("Generating monthly timeline..."):
                monthly_timeline_fig = analyzer.monthly_timeline('Overall', df)
            st.plotly_chart(monthly_timeline_fig, use_container_width=True)

            st.subheader("Daily Message Timeline")
            with st.spinner("Generating daily timeline..."):
                daily_timeline_fig = analyzer.daily_timeline('Overall', df)
            st.plotly_chart(daily_timeline_fig, use_container_width=True)

        # 9. Activity Heatmap (Original - Returns DataFrame)
        with st.expander("9. Activity Heatmap (Table)", expanded=False):
            st.subheader("Activity Patterns (Day vs. Period)")
            with st.spinner("Generating activity heatmap data..."):
                activity_heatmap_df = analyzer.activity_heatmap('Overall', df)
            st.dataframe(activity_heatmap_df)
            # Optional: Plotting the heatmap requires seaborn
            try:
                import seaborn as sns
                fig_heatmap, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(activity_heatmap_df, cmap="YlGnBu", ax=ax)
                plt.title('Activity Heatmap (Day vs Period)')
                st.pyplot(fig_heatmap)
            except ImportError:
                 st.info("Install seaborn (`pip install seaborn`) to view the heatmap plot.")
            except Exception as e:
                 st.warning(f"Could not plot heatmap: {e}")


        # 10. Week Activity Map (Plotly Heatmap)
        # with st.expander("10. Weekly Activity Map (Plotly Heatmap)", expanded=False):
        #     st.subheader("Weekly Activity Pattern")
        #     with st.spinner("Generating weekly activity map..."):
        #         week_activity_fig = analyzer.week_activity_map('Overall', df)
        #     st.plotly_chart(week_activity_fig, use_container_width=True)

        # 11. Month Activity Map (Bar Chart)
        with st.expander("11. Monthly Activity Map (Bar Chart)", expanded=False):
            st.subheader("Monthly Activity Pattern")
            with st.spinner("Generating monthly activity map..."):
                month_activity_fig = analyzer.month_activity_map('Overall', df)
            st.plotly_chart(month_activity_fig, use_container_width=True)

        # 12. Busiest Hours Analysis (Returns Series)
        with st.expander("12. Busiest Hours Analysis", expanded=False):
            st.subheader("Message Count per Hour")
            with st.spinner("Analyzing busiest hours..."):
                busiest_hours_series = analyzer.busiest_hours_analysis(df)
                busiest_hours_df = busiest_hours_series.reset_index()
                busiest_hours_df.columns = ['Hour', 'Message Count']
            st.dataframe(busiest_hours_df)
            # Optional: Plot
            try:
                 fig_busy_hours = go.Figure(data=[go.Bar(x=busiest_hours_df['Hour'], y=busiest_hours_df['Message Count'])])
                 fig_busy_hours.update_layout(title="Message Count by Hour of Day", xaxis_title="Hour", yaxis_title="Message Count")
                 st.plotly_chart(fig_busy_hours, use_container_width=True)
            except Exception as e:
                 st.warning(f"Could not plot busiest hours: {e}")


        # 13. Sentiment Percentage Calculation
        with st.expander("13. Sentiment Percentage Calculation", expanded=False):
            st.subheader("User Sentiment Percentages")
            with st.spinner("Calculating sentiment percentages..."):
                sentiments, most_positive, most_negative = analyzer.calculate_sentiment_percentage('Overall', df)
                sentiment_list = []
                for user, scores in sentiments.items():
                     sentiment_list.append({'User': user, 'Positive %': scores[0], 'Negative %': scores[1]})
                sentiment_df = pd.DataFrame(sentiment_list)
            st.dataframe(sentiment_df)
            col1, col2 = st.columns(2)
            col1.metric("Most Positive User", most_positive)
            col2.metric("Most Negative User", most_negative)
            # Optional Plot
            try:
                sentiment_df_melt = sentiment_df.melt(id_vars='User', var_name='Sentiment Type', value_name='Percentage')
                fig_sent_perc = px.bar(sentiment_df_melt, x='User', y='Percentage', color='Sentiment Type', 
                                       barmode='group', title='Positive vs. Negative Sentiment Percentage per User')
                st.plotly_chart(fig_sent_perc, use_container_width=True)
            except NameError: # If px (plotly.express) is not imported
                st.info("Plotly Express needed for sentiment percentage bar chart.")
            except Exception as e:
                 st.warning(f"Could not plot sentiment percentages: {e}")

        # 14. Analyze and Plot Sentiment (Distribution and Trend)
        with st.expander("14. Sentiment Distribution and Trend", expanded=False):
            st.subheader("Overall Sentiment Analysis")
            with st.spinner("Analyzing sentiment distribution and trend..."):
                dist_fig, trend_fig = analyzer.analyze_and_plot_sentiment('Overall', df)
            st.plotly_chart(dist_fig, use_container_width=True)
            st.plotly_chart(trend_fig, use_container_width=True)

        # 15. Calculate Monthly Sentiment Trend
        with st.expander("15. Monthly Sentiment Trend", expanded=False):
            st.subheader("Sentiment Trend Over Months")
            with st.spinner("Calculating monthly sentiment trend..."):
                monthly_sentiment_fig = analyzer.calculate_monthly_sentiment_trend(df)
            st.plotly_chart(monthly_sentiment_fig, use_container_width=True)

        # 16. Message Count Aggregated Graph
        with st.expander("16. Message Count Distribution (Pie Chart)", expanded=False):
            st.subheader("Total Messages per User")
            with st.spinner("Aggregating message counts..."):
                msg_count_fig, most_messages_winner = analyzer.message_count_aggregated_graph(df)
            st.metric("User with Most Messages", most_messages_winner)
            st.plotly_chart(msg_count_fig, use_container_width=True)

        # 17. Conversation Starter Graph
        with st.expander("17. Conversation Starters (Pie Chart)", expanded=False):
            st.subheader("Who Starts Conversations Most Often?")
            with st.spinner("Analyzing conversation starters..."):
                convo_starter_fig, most_frequent_starter = analyzer.conversation_starter_graph(df)
            st.metric("Most Frequent Conversation Starter", most_frequent_starter)
            st.plotly_chart(convo_starter_fig, use_container_width=True)

        # 18. Conversation Size Aggregated Graph
        with st.expander("18. Weekly Average Conversation Size", expanded=False):
            st.subheader("Average Conversation Size Over Time")
            with st.spinner("Calculating conversation sizes..."):
                convo_size_fig = analyzer.conversation_size_aggregated_graph(df)
            st.plotly_chart(convo_size_fig, use_container_width=True)

        # 19. Calculate Average Late Reply Time
        with st.expander("19. Average Late Reply Time Analysis", expanded=False):
            st.subheader("Late Reply Analysis (Replies > 24 hours)")
            with st.spinner("Calculating late reply times..."):
                 try:
                    late_reply_fig, avg_late_reply_df, overall_avg_late_reply = analyzer.calculate_average_late_reply_time(df, threshold_hours=24)
                    st.metric("Overall Average Late Reply Time (hours)", f"{overall_avg_late_reply:.2f}")
                    st.dataframe(avg_late_reply_df)
                    st.pyplot(late_reply_fig)
                 except Exception as e:
                    st.warning(f"Could not perform late reply analysis: {e}")


        # 20. Reply Pattern Analysis (Longest Reply)
        with st.expander("20. Longest Reply Time", expanded=False):
            st.subheader("Longest Wait for a Reply")
            with st.spinner("Finding longest reply..."):
                try:
                    user_longest, time_longest, msg_longest, reply_original = analyzer.analyze_reply_patterns(df)
                    st.text(f"User with longest reply time: {user_longest}")
                    st.text(f"Reply took {time_longest:.2f} minutes")
                    with st.expander("Original Message"):
                        st.text(reply_original)
                    with st.expander("Reply Message"):
                        st.text(msg_longest)
                except Exception as e:
                    st.warning(f"Could not analyze reply patterns: {e}")

        # 21. Message Types Analysis
        with st.expander("21. Message Types Analysis", expanded=False):
            st.subheader("Distribution of Message Types")
            with st.spinner("Analyzing message types..."):
                 message_types = analyzer.analyze_message_types('Overall', df)
                 message_types_df = pd.DataFrame(message_types.items(), columns=['Type', 'Count'])
            st.dataframe(message_types_df)
            # Optional: Create and show the chart
            try:
                message_types_fig = analyzer.create_message_types_chart(message_types) # Assumes this returns a Plotly fig
                st.plotly_chart(message_types_fig, use_container_width=True)
            except AttributeError: # If create_message_types_chart doesn't exist or isn't implemented
                 st.info("Bar chart visualization for message types is not implemented yet.")
                 try: # Attempt basic plotly pie chart
                     import plotly.express as px
                     fig_pie = px.pie(message_types_df, values='Count', names='Type', title='Message Type Distribution')
                     st.plotly_chart(fig_pie, use_container_width=True)
                 except ImportError:
                     st.info("Install plotly (`pip install plotly`) for a pie chart visualization.")
                 except Exception as e:
                     st.warning(f"Could not create pie chart: {e}")
            except Exception as e:
                 st.warning(f"Could not create message types chart: {e}")

        # 22. Conversation Pattern Analysis
        with st.expander("22. Conversation Pattern Analysis", expanded=False):
            st.subheader("General Conversation Metrics")
            with st.spinner("Analyzing conversation patterns..."):
                conversation_patterns = analyzer.analyze_conversation_patterns(df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Conversations", conversation_patterns['total_conversations'])
            col2.metric("Avg Length (Messages)", f"{conversation_patterns['avg_conversation_length']:.2f}")
            col3.metric("Avg Duration (Minutes)", f"{conversation_patterns['avg_conversation_duration_mins']:.2f}")

            st.subheader("Conversation Starters")
            starters_df = pd.DataFrame(conversation_patterns['conversation_starters'].items(), columns=['User', 'Count']).sort_values('Count', ascending=False)
            st.dataframe(starters_df)

            # Optional: Create and show charts
            try:
                convo_pattern_charts = analyzer.create_conversation_patterns_chart(conversation_patterns) # Assumes this returns dict of figs
                if 'conversation_starters' in convo_pattern_charts:
                    st.plotly_chart(convo_pattern_charts['conversation_starters'], use_container_width=True)
                if 'conversation_metrics' in convo_pattern_charts:
                     st.plotly_chart(convo_pattern_charts['conversation_metrics'], use_container_width=True)
            except AttributeError:
                 st.info("Chart visualization for conversation patterns is not implemented yet.")
                 try: # Attempt basic plotly pie chart for starters
                     import plotly.express as px
                     fig_pie = px.pie(starters_df, values='Count', names='User', title='Conversation Starters')
                     st.plotly_chart(fig_pie, use_container_width=True)
                 except ImportError:
                      st.info("Install plotly (`pip install plotly`) for a starter pie chart visualization.")
                 except Exception as e:
                     st.warning(f"Could not create starter pie chart: {e}")
            except Exception as e:
                 st.warning(f"Could not create conversation pattern charts: {e}")


        # 23. User Interaction Analysis
        with st.expander("23. User Interaction Analysis", expanded=False):
            st.subheader("User Interaction Network")
            with st.spinner("Analyzing user interactions..."):
                try:
                    G, interactions = analyzer.analyze_user_interactions(df)
                    st.text(f"Number of users in interaction graph: {len(G.nodes())}")
                    st.text(f"Number of interactions (edges): {len(G.edges())}")
                    # Optional: Create and show graph
                    interaction_graph_fig = analyzer.create_user_interaction_graph(G)
                    st.plotly_chart(interaction_graph_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate interaction graph: {e}")

        # 24. Time Pattern Analysis (DataFrames)
        with st.expander("24. Time Pattern Analysis", expanded=False):
            st.subheader("Detailed Activity Patterns")
            with st.spinner("Analyzing time patterns..."):
                time_patterns = analyzer.analyze_time_patterns(df)

            st.text("Hourly activity summary:")
            st.dataframe(time_patterns['hourly_activity'])
            
            st.text("Daily activity summary:")
            st.dataframe(time_patterns['daily_activity'])

            st.text("Monthly activity summary:")
            st.dataframe(time_patterns['monthly_activity'])

            st.text("User hourly activity summary:")
            st.dataframe(time_patterns['user_hourly'])

            st.text("User daily activity summary:")
            st.dataframe(time_patterns['user_daily'])

            # Optional Plots
            try:
                import plotly.express as px
                fig_h = px.bar(time_patterns['hourly_activity'], x='hour', y='message_count', title='Messages per Hour')
                st.plotly_chart(fig_h, use_container_width=True)
                fig_d = px.bar(time_patterns['daily_activity'], x='day_name', y='message_count', title='Messages per Day of Week')
                st.plotly_chart(fig_d, use_container_width=True)
                fig_m = px.bar(time_patterns['monthly_activity'], x='month', y='message_count', title='Messages per Month')
                st.plotly_chart(fig_m, use_container_width=True)
            except ImportError:
                st.info("Install plotly (`pip install plotly`) for time pattern charts.")
            except Exception as e:
                st.warning(f"Could not create time pattern charts: {e}")


        # 25. Message Length Analysis
        with st.expander("25. Message Length Analysis", expanded=False):
            st.subheader("Message Length Statistics")
            with st.spinner("Analyzing message lengths..."):
                 message_length = analyzer.analyze_message_length(df)

            col1, col2, col3 = st.columns(3)
            col1.metric("Average Length (Words)", f"{message_length['avg_length']:.2f}")
            col2.metric("Maximum Length (Words)", message_length['max_length'])
            col3.metric("Minimum Length (Words)", message_length['min_length'])

            st.text(f"Longest message by: {message_length['longest_message']['user']} at {message_length['longest_message']['date']} (Length: {message_length['longest_message']['length']})")
            with st.expander("Show Longest Message Text"):
                st.text(message_length['longest_message']['message'])
                
            st.text("Average message length per user:")
            st.dataframe(message_length['user_avg_length'])
            
            # Optional Plot: Histogram of message lengths
            try:
                 import plotly.express as px
                 fig_len_hist = px.histogram(df, x="Message Length", title="Distribution of Message Lengths (Words)")
                 st.plotly_chart(fig_len_hist, use_container_width=True)
                 
                 fig_len_user = px.bar(message_length['user_avg_length'], x='user', y='Message Length', title='Average Message Length per User')
                 st.plotly_chart(fig_len_user, use_container_width=True)
                 
            except Exception as e:
                 st.warning(f"Could not plot message length distributions: {e}")


        # 26. Response Time Analysis
        with st.expander("26. Response Time Analysis", expanded=False):
             st.subheader("Reply Response Times")
             with st.spinner("Analyzing response times..."):
                 try:
                     response_times = analyzer.analyze_response_times(df)
                     col1, col2 = st.columns(2)
                     col1.metric("Average Response Time (Minutes)", f"{response_times.get('avg_response_time', 'N/A'):.2f}" if isinstance(response_times.get('avg_response_time'), (int, float)) else 'N/A')
                     col2.metric("Median Response Time (Minutes)", f"{response_times.get('median_response_time', 'N/A'):.2f}" if isinstance(response_times.get('median_response_time'), (int, float)) else 'N/A')
                     
                     if 'user_response_times' in response_times:
                         st.text("User average response times (Minutes):")
                         st.dataframe(response_times['user_response_times'])
                     
                     # Optional Plot
                     if 'response_time_data' in response_times:
                          import plotly.express as px
                          fig_resp_hist = px.histogram(response_times['response_time_data'], x='Response Time (min)', title='Distribution of Response Times')
                          st.plotly_chart(fig_resp_hist, use_container_width=True)
                          
                          if 'user_response_times' in response_times:
                              fig_resp_user = px.bar(response_times['user_response_times'].reset_index(), x='user', y='mean', title='Average Response Time per User (Minutes)')
                              st.plotly_chart(fig_resp_user, use_container_width=True)
                              
                 except Exception as e:
                     st.warning(f"Could not analyze response times: {e}")


        # 27. Topic Modeling
        with st.expander("27. Topic Modeling", expanded=False):
            st.subheader("Identified Conversation Topics")
            num_topics = st.slider("Number of Topics", 2, 10, 3)
            num_words = st.slider("Number of Words per Topic", 3, 15, 5)
            with st.spinner(f"Running Topic Modeling (k={num_topics})..."):
                topics = analyzer.analyze_topic_modeling(df, num_topics=num_topics, num_words=num_words)

            if topics['success']:
                st.text("Topics extracted:")
                for topic in topics['topics']:
                    st.markdown(f"*   **Topic {topic['topic_id']}**: {', '.join(topic['words'])}")
                
                st.text("Topic distribution per message (Top 10 messages):")
                st.dataframe(topics['topic_counts'].head(10))
                
                # Optional Plot
                try:
                    import plotly.express as px
                    topic_dist = topics['topic_counts']['Dominant Topic'].value_counts().reset_index()
                    topic_dist.columns = ['Topic ID', 'Number of Messages']
                    fig_topic_dist = px.bar(topic_dist, x='Topic ID', y='Number of Messages', title='Number of Messages per Topic')
                    st.plotly_chart(fig_topic_dist, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot topic distribution: {e}")

            else:
                st.error(f"Topic modeling failed: {topics.get('error', 'Unknown error')}")


        # 28. Emoji Usage Analysis (Detailed)
        with st.expander("28. Emoji Usage Analysis (Detailed)", expanded=False):
            st.subheader("Detailed Emoji Statistics")
            with st.spinner("Analyzing emoji usage..."):
                try:
                    emoji_usage = analyzer.analyze_emoji_usage(df)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Emojis Used", emoji_usage['total_emojis'])
                    col2.metric("Unique Emojis Used", emoji_usage['emoji_diversity'])
                    col3.metric("Emojis per Message", f"{emoji_usage['emoji_density']:.2f}")

                    if not emoji_usage['emoji_counts'].empty:
                        st.text("Top 10 emojis overall:")
                        st.dataframe(emoji_usage['emoji_counts'].head(10))
                        
                    if not emoji_usage['user_emoji_counts'].empty:
                        st.text("Top 5 emojis per user:")
                        st.dataframe(emoji_usage['user_emoji_counts'])

                    if not emoji_usage['emoji_sentiment'].empty:
                        st.text("Average sentiment per emoji (Top 10 most frequent):")
                        st.dataframe(emoji_usage['emoji_sentiment'].head(10))

                    # Optional Plots
                    try:
                        import plotly.express as px
                        if not emoji_usage['emoji_counts'].empty:
                            fig_top_emojis = px.bar(emoji_usage['emoji_counts'].head(20).reset_index(), x='Emoji', y='Count', title='Top 20 Most Used Emojis')
                            st.plotly_chart(fig_top_emojis, use_container_width=True)
                        
                        if not emoji_usage['user_emoji_pivot'].empty:
                            fig_user_emoji_heatmap = px.imshow(emoji_usage['user_emoji_pivot'], title='Emoji Usage Heatmap (User vs Emoji)', aspect="auto")
                            st.plotly_chart(fig_user_emoji_heatmap, use_container_width=True)
                            
                    except Exception as e:
                        st.warning(f"Could not plot detailed emoji usage: {e}")
                        
                except Exception as e:
                    st.warning(f"Could not perform detailed emoji analysis: {e}")
        
        # 29. Sentiment Trend Analysis (Detailed)
        with st.expander("29. Sentiment Trend Analysis (Detailed)", expanded=False):
            st.subheader("Detailed Sentiment Analysis")
            with st.spinner("Analyzing sentiment trends..."):
                 try:
                     sentiment_trends = analyzer.analyze_sentiment_trends(df)
                     
                     st.text("Sentiment distribution counts:")
                     sentiment_counts_df = pd.DataFrame(sentiment_trends['sentiment_counts'].items(), columns=['Sentiment', 'Count'])
                     st.dataframe(sentiment_counts_df)
                     
                     st.metric("Average Sentiment Score", f"{sentiment_trends['avg_sentiment']:.2f}")
                     
                     st.text("User average sentiment:")
                     st.dataframe(sentiment_trends['user_sentiment'])
                     
                     st.text("Sentiment Trend Over Time (Monthly Aggregation):")
                     st.dataframe(sentiment_trends['monthly_sentiment_trend'])

                     # Optional Plots
                     try:
                         import plotly.express as px
                         fig_sent_dist = px.pie(sentiment_counts_df, values='Count', names='Sentiment', title='Overall Sentiment Distribution')
                         st.plotly_chart(fig_sent_dist, use_container_width=True)
                         
                         fig_user_sent = px.bar(sentiment_trends['user_sentiment'].reset_index(), x='user', y='sentiment_score', title='Average Sentiment Score per User')
                         st.plotly_chart(fig_user_sent, use_container_width=True)
                         
                         fig_sent_time = px.line(sentiment_trends['monthly_sentiment_trend'].reset_index(), x='month_year', y='sentiment_score', title='Sentiment Score Trend Over Time')
                         st.plotly_chart(fig_sent_time, use_container_width=True)
                         
                     except Exception as e:
                         st.warning(f"Could not plot detailed sentiment trends: {e}")

                 except Exception as e:
                    st.warning(f"Could not perform detailed sentiment trend analysis: {e}")

        # 30. Word Usage Analysis (Detailed)
        with st.expander("30. Word Usage Analysis (Detailed)", expanded=False):
            st.subheader("Detailed Word Usage Statistics")
            with st.spinner("Analyzing word usage..."):
                 try:
                     word_usage = analyzer.analyze_word_usage(df)
                     col1, col2, col3 = st.columns(3)
                     col1.metric("Total Words (Filtered)", word_usage['total_words'])
                     col2.metric("Unique Words (Filtered)", word_usage['word_diversity'])
                     col3.metric("Avg Words per Message (Filtered)", f"{word_usage['words_per_message']:.2f}")

                     if not word_usage['word_counts'].empty:
                         st.text("Top 20 most frequent words (filtered):")
                         st.dataframe(word_usage['word_counts'].head(20))
                         
                     if 'user_word_counts' in word_usage and not word_usage['user_word_counts'].empty:
                         st.text("Top 10 words per user:")
                         st.dataframe(word_usage['user_word_counts'])

                     # Optional Plots
                     try:
                         import plotly.express as px
                         if not word_usage['word_counts'].empty:
                              fig_top_words = px.bar(word_usage['word_counts'].head(25).reset_index(), x='Word', y='Count', title='Top 25 Most Frequent Words')
                              st.plotly_chart(fig_top_words, use_container_width=True)
                              
                         # Consider adding a plot for user word diversity or common unique words if available
                         
                     except Exception as e:
                         st.warning(f"Could not plot detailed word usage: {e}")

                 except Exception as e:
                     st.warning(f"Could not perform detailed word usage analysis: {e}")

        # 31. Conversation Flow Analysis (Detailed)
        with st.expander("31. Conversation Flow Analysis (Detailed)", expanded=False):
             st.subheader("Detailed Conversation Flow Metrics")
             with st.spinner("Analyzing conversation flow..."):
                 try:
                     conversation_flow = analyzer.analyze_conversation_flow(df)
                     st.metric("Total Conversations Identified", conversation_flow['total_conversations'])

                     st.text("Conversation Statistics (Sample):")
                     st.dataframe(conversation_flow['conversation_stats'].head())

                     st.text("Conversation Starters:")
                     st.dataframe(conversation_flow['conversation_starters'])

                     st.text("Conversation Enders:")
                     st.dataframe(conversation_flow['conversation_enders'])
                     
                     # Optional Plots
                     try:
                         import plotly.express as px
                         fig_conv_len = px.histogram(conversation_flow['conversation_stats'], x='length', title='Distribution of Conversation Lengths (Messages)')
                         st.plotly_chart(fig_conv_len, use_container_width=True)
                         
                         fig_conv_dur = px.histogram(conversation_flow['conversation_stats'], x='duration_minutes', title='Distribution of Conversation Durations (Minutes)')
                         st.plotly_chart(fig_conv_dur, use_container_width=True)
                         
                         fig_starters = px.pie(conversation_flow['conversation_starters'], values='count', names='user', title='Conversation Starters')
                         st.plotly_chart(fig_starters, use_container_width=True)

                         fig_enders = px.pie(conversation_flow['conversation_enders'], values='count', names='user', title='Conversation Enders')
                         st.plotly_chart(fig_enders, use_container_width=True)
                         
                     except Exception as e:
                         st.warning(f"Could not plot conversation flow details: {e}")

                 except Exception as e:
                    st.warning(f"Could not perform detailed conversation flow analysis: {e}")
                    
        # 33. Conversation Mood Shifts (Detailed)
        with st.expander("33. Conversation Mood Shifts (Detailed)", expanded=False):
            st.subheader("Analysis of Mood Changes within Conversations")
            with st.spinner("Analyzing mood shifts..."):
                try:
                    mood_shifts = analyzer.analyze_conversation_mood_shifts(df)
                    
                    if mood_shifts['mood_lifters']:
                        st.text("Top Mood Lifters (Users increasing positivity):")
                        mood_lifters_df = pd.DataFrame(mood_shifts['mood_lifters'], columns=['User', 'Lift Count', 'Lift Ratio'])
                        st.dataframe(mood_lifters_df)

                    if mood_shifts['mood_dampeners']:
                        st.text("Top Mood Dampeners (Users decreasing positivity):")
                        mood_dampeners_df = pd.DataFrame(mood_shifts['mood_dampeners'], columns=['User', 'Dampen Count', 'Dampen Ratio'])
                        st.dataframe(mood_dampeners_df)

                    if mood_shifts['top_shifters']:
                        st.text("Top Overall Mood Shifters (by number of shifts):")
                        top_shifters_df = pd.DataFrame(mood_shifts['top_shifters'], columns=['User', 'Shift Count', 'Shift Ratio', 'Avg Shift'])
                        st.dataframe(top_shifters_df)
                        
                    # Optional Plots (Could plot distributions of shift scores, or bar charts for top users)
                    try:
                        import plotly.express as px
                        if not mood_shifts['shift_data'].empty:
                            fig_shift_hist = px.histogram(mood_shifts['shift_data'], x='sentiment_shift', title='Distribution of Sentiment Shifts Between Messages')
                            st.plotly_chart(fig_shift_hist, use_container_width=True)
                            
                        if mood_shifts['mood_lifters']:
                            fig_lifters = px.bar(mood_lifters_df.head(10), x='User', y='Lift Count', title='Top Mood Lifters')
                            st.plotly_chart(fig_lifters, use_container_width=True)

                        if mood_shifts['mood_dampeners']:
                            fig_dampeners = px.bar(mood_dampeners_df.head(10), x='User', y='Dampen Count', title='Top Mood Dampeners')
                            st.plotly_chart(fig_dampeners, use_container_width=True)

                    except Exception as e:
                        st.warning(f"Could not plot mood shift details: {e}")

                except Exception as e:
                    st.warning(f"Could not perform mood shift analysis: {e}")

    except ValueError as e:
        st.error(f"Error preprocessing chat file: {e}. Please ensure it's a valid WhatsApp export.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else:
    st.info("Awaiting chat file upload.")

st.sidebar.header("About")
st.sidebar.info("This application analyzes WhatsApp chat exports using the `whatsapp-reality` library.")
st.sidebar.info("Developed as a demonstration.") 