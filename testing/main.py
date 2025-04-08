from whatsapp_reality import preprocess, analyzer
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Read and preprocess your chat file
with open('ios_12hr.txt', 'r', encoding='utf-8') as file:
    chat_data = file.read()
    
# Create the DataFrame
df = preprocess(chat_data)

print("WhatsApp Chat Analysis - Comprehensive Test")
print("=" * 60)

# 1. Basic Statistics
print("\n1. BASIC STATISTICS")
print("-" * 50)
messages, words, media, links = analyzer.fetch_stats('Overall', df)
print(f"Total Messages: {messages}")
print(f"Total Words: {words}")
print(f"Media Messages: {media}")
print(f"Links Shared: {links}")

# 2. User Activity Analysis
print("\n2. USER ACTIVITY ANALYSIS (Most Busy Users)")
print("-" * 50)
fig_busy_users, df_percent = analyzer.most_busy_users(df)
print("Most Active Users (% of total messages):")
print(df_percent.head())
# fig_busy_users.show() # Uncomment to display figure

# 3. Plotly Word Cloud Generation
print("\n3. PLOTLY WORD CLOUD GENERATION")
print("-" * 50)
plotly_wordcloud_fig = analyzer.create_plotly_wordcloud('Overall', df,True)
print("Plotly word cloud generated.")
# plotly_wordcloud_fig.show() # Uncomment to display figure

# 4. Matplotlib Word Cloud Generation (Original)
print("\n4. MATPLOTLIB WORD CLOUD GENERATION")
print("-" * 50)
wordcloud_obj = analyzer.create_wordcloud('Overall', df)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_obj)
plt.axis('off')
plt.title("Matplotlib Word Cloud")
plt.savefig('wordcloud.png')
print("Matplotlib Word cloud generated and saved as 'wordcloud.png'")
# plt.show() # Uncomment to display figure

# 5. Most Common Words
print("\n5. MOST COMMON WORDS")
print("-" * 50)
common_words_fig = analyzer.most_common_words('Overall', df)
print("Most common words analysis completed.")
# common_words_fig.show() # Uncomment to display figure

# 6. Emoji Analysis (Original Pie Chart)
print("\n6. EMOJI ANALYSIS (Original Pie Chart)")
print("-" * 50)
emoji_fig = analyzer.emoji_analysis('Overall', df)
print("Original emoji analysis completed.")
# emoji_fig.show() # Uncomment to display figure

# 7. Emoji Helper (Similar Pie Chart)
print("\n7. EMOJI HELPER")
print("-" * 50)
emoji_helper_fig = analyzer.emoji_helper('Overall', df)
print("Emoji helper analysis completed.")
# emoji_helper_fig.show() # Uncomment to display figure

# 8. Timeline Analysis
print("\n8. TIMELINE ANALYSIS")
print("-" * 50)
monthly_timeline_fig = analyzer.monthly_timeline('Overall', df)
daily_timeline_fig = analyzer.daily_timeline('Overall', df)
print("Monthly and Daily timeline analysis completed.")
# monthly_timeline_fig.show() # Uncomment to display figure
# daily_timeline_fig.show() # Uncomment to display figure

# 9. Activity Heatmap (Original - Returns DataFrame)
print("\n9. ACTIVITY HEATMAP (DataFrame Output)")
print("-" * 50)
activity_heatmap_df = analyzer.activity_heatmap('Overall', df)
print("Activity patterns by day and period (DataFrame):")
print(activity_heatmap_df)

# 10. Week Activity Map (Plotly Heatmap)
print("\n10. WEEK ACTIVITY MAP (Plotly Heatmap)")
print("-" * 50)
week_activity_fig = analyzer.week_activity_map('Overall', df)
print("Week activity map generated.")
# week_activity_fig.show() # Uncomment to display figure

# 11. Month Activity Map (Bar Chart)
print("\n11. MONTH ACTIVITY MAP (Bar Chart)")
print("-" * 50)
month_activity_fig = analyzer.month_activity_map('Overall', df)
print("Month activity map generated.")
# month_activity_fig.show() # Uncomment to display figure

# 12. Busiest Hours Analysis (Returns Series)
print("\n12. BUSIEST HOURS ANALYSIS (Series Output)")
print("-" * 50)
busiest_hours_series = analyzer.busiest_hours_analysis(df)
print("Message count per hour:")
print(busiest_hours_series.sort_index())

# 13. Sentiment Percentage Calculation
print("\n13. SENTIMENT PERCENTAGE CALCULATION")
print("-" * 50)
sentiments, most_positive, most_negative = analyzer.calculate_sentiment_percentage('Overall', df)
print("User Sentiment Percentages (Positive%, Negative%):")
# Print first 5 users for brevity
for user, scores in list(sentiments.items())[:5]:
    print(f"  {user}: {scores}")
print(f"Most positive user: {most_positive}")
print(f"Most negative user: {most_negative}")

# 14. Analyze and Plot Sentiment (Distribution and Trend)
print("\n14. ANALYZE AND PLOT SENTIMENT")
print("-" * 50)
dist_fig, trend_fig = analyzer.analyze_and_plot_sentiment('Overall', df)
print("Sentiment distribution and trend plots generated.")
# dist_fig.show() # Uncomment to display figure
# trend_fig.show() # Uncomment to display figure

# 15. Calculate Monthly Sentiment Trend
print("\n15. CALCULATE MONTHLY SENTIMENT TREND")
print("-" * 50)
monthly_sentiment_fig = analyzer.calculate_monthly_sentiment_trend(df)
print("Monthly sentiment trend plot generated.")
# monthly_sentiment_fig.show() # Uncomment to display figure

# 16. Message Count Aggregated Graph
print("\n16. MESSAGE COUNT AGGREGATED GRAPH")
print("-" * 50)
msg_count_fig, most_messages_winner = analyzer.message_count_aggregated_graph(df)
print(f"User who sent the most messages: {most_messages_winner}")
print("Message count pie chart generated.")
# msg_count_fig.show() # Uncomment to display figure

# 17. Conversation Starter Graph
print("\n17. CONVERSATION STARTER GRAPH")
print("-" * 50)
convo_starter_fig, most_frequent_starter = analyzer.conversation_starter_graph(df)
print(f"User who started the most conversations: {most_frequent_starter}")
print("Conversation starter pie chart generated.")
# convo_starter_fig.show() # Uncomment to display figure

# 18. Conversation Size Aggregated Graph
print("\n18. CONVERSATION SIZE AGGREGATED GRAPH")
print("-" * 50)
convo_size_fig = analyzer.conversation_size_aggregated_graph(df)
print("Weekly average conversation size graph generated.")
# convo_size_fig.show() # Uncomment to display figure

# 19. Calculate Average Late Reply Time
print("\n19. AVERAGE LATE REPLY TIME ANALYSIS")
print("-" * 50)
late_reply_fig, avg_late_reply_df, overall_avg_late_reply = analyzer.calculate_average_late_reply_time(df, threshold_hours=24) # Using 24hr threshold for test
print(f"Overall average late reply time: {overall_avg_late_reply:.2f} hours")
print("Average late reply times by user (DataFrame):")
print(avg_late_reply_df.head())
print("Late reply time plot generated.")
# late_reply_fig.show() # Uncomment to display figure

# 20. Reply Pattern Analysis (Longest Reply)
print("\n20. REPLY PATTERN ANALYSIS (Longest Reply)")
print("-" * 50)
user_longest, time_longest, msg_longest, reply_original = analyzer.analyze_reply_patterns(df)
print(f"User with longest reply time: {user_longest}")
print(f"Reply took {time_longest:.2f} minutes")
# print(f"Original message: {reply_original}") # Often long, commented out
# print(f"Reply message: {msg_longest}")      # Often long, commented out

# 21. Message Types Analysis
print("\n21. MESSAGE TYPES ANALYSIS")
print("-" * 50)
message_types = analyzer.analyze_message_types('Overall', df)
print("Message type distribution:")
for msg_type, count in message_types.items():
    print(f"  {msg_type}: {count}")
# Optional: Create and show the chart
# message_types_fig = analyzer.create_message_types_chart(message_types)
# message_types_fig.show()

# 22. Conversation Pattern Analysis
print("\n22. CONVERSATION PATTERN ANALYSIS")
print("-" * 50)
conversation_patterns = analyzer.analyze_conversation_patterns(df)
print(f"Total conversations: {conversation_patterns['total_conversations']}")
print(f"Average conversation length: {conversation_patterns['avg_conversation_length']:.2f} messages")
print(f"Average conversation duration: {conversation_patterns['avg_conversation_duration_mins']:.2f} minutes")
print("Conversation starters (Top 5):")
starters = sorted(conversation_patterns['conversation_starters'].items(), key=lambda item: item[1], reverse=True)
for user, count in starters[:5]:
    print(f"  {user}: {count}")
# Optional: Create and show charts
# convo_pattern_charts = analyzer.create_conversation_patterns_chart(conversation_patterns)
# if 'conversation_starters' in convo_pattern_charts:
#     convo_pattern_charts['conversation_starters'].show()
# if 'conversation_metrics' in convo_pattern_charts:
#     convo_pattern_charts['conversation_metrics'].show()

# 23. User Interaction Analysis
print("\n23. USER INTERACTION ANALYSIS")
print("-" * 50)
G, interactions = analyzer.analyze_user_interactions(df)
print(f"Number of users in interaction graph: {len(G.nodes())}")
print(f"Number of interactions (edges): {len(G.edges())}")
# Optional: Create and show graph
# interaction_graph_fig = analyzer.create_user_interaction_graph(G)
# interaction_graph_fig.show()

# 24. Time Pattern Analysis (DataFrames)
print("\n24. TIME PATTERN ANALYSIS (DataFrames)")
print("-" * 50)
time_patterns = analyzer.analyze_time_patterns(df)
print("Hourly activity summary (first 5 rows):")
print(time_patterns['hourly_activity'].head())
print("\nDaily activity summary:")
print(time_patterns['daily_activity'])
print("\nMonthly activity summary:")
print(time_patterns['monthly_activity'])
print("\nUser hourly activity summary (first 5 rows):")
print(time_patterns['user_hourly'].head())
print("\nUser daily activity summary (first 5 rows):")
print(time_patterns['user_daily'].head())

# 25. Message Length Analysis
print("\n25. MESSAGE LENGTH ANALYSIS")
print("-" * 50)
message_length = analyzer.analyze_message_length(df)
print(f"Average message length: {message_length['avg_length']:.2f} words")
print(f"Maximum message length: {message_length['max_length']} words")
print(f"Minimum message length: {message_length['min_length']} words")
print(f"Longest message by: {message_length['longest_message']['user']} at {message_length['longest_message']['date']} (Length: {message_length['longest_message']['length']})")

# 26. Response Time Analysis
print("\n26. RESPONSE TIME ANALYSIS")
print("-" * 50)
response_times = analyzer.analyze_response_times(df)
print(f"Average response time: {response_times.get('avg_response_time', 'N/A'):.2f} minutes")
print(f"Median response time: {response_times.get('median_response_time', 'N/A'):.2f} minutes")
print("User average response times (Top 5):")
if 'user_response_times' in response_times:
    print(response_times['user_response_times'].sort_values('mean').head())

# 27. Topic Modeling
print("\n27. TOPIC MODELING")
print("-" * 50)
topics = analyzer.analyze_topic_modeling(df, num_topics=3, num_words=5)
if topics['success']:
    print("Topics extracted:")
    for topic in topics['topics']:
        print(f"  Topic {topic['topic_id']}: {', '.join(topic['words'])}")
    print("\nTopic counts (Top 5 messages):")
    print(topics['topic_counts'].head())
else:
    print("Topic modeling failed:", topics.get('error', 'Unknown error'))

# 28. Emoji Usage Analysis (Detailed)
print("\n28. EMOJI USAGE ANALYSIS (Detailed)")
print("-" * 50)
emoji_usage = analyzer.analyze_emoji_usage(df)
print(f"Total emojis used: {emoji_usage['total_emojis']}")
print(f"Emoji diversity: {emoji_usage['emoji_diversity']} unique emojis")
print(f"Emoji density: {emoji_usage['emoji_density']:.2f} emojis per message")
if not emoji_usage['emoji_counts'].empty:
    print("Top 5 emojis overall:")
    print(emoji_usage['emoji_counts'].head())
    
# 29. Sentiment Trend Analysis (Detailed)
print("\n29. SENTIMENT TREND ANALYSIS (Detailed)")
print("-" * 50)
sentiment_trends = analyzer.analyze_sentiment_trends(df)
print("Sentiment distribution counts:")
for sentiment, count in sentiment_trends['sentiment_counts'].items():
    print(f"  {sentiment}: {count}")
print(f"Average sentiment score: {sentiment_trends['avg_sentiment']:.2f}")
print("\nUser average sentiment (Top 5 most positive):")
print(sentiment_trends['user_sentiment'].sort_values('sentiment_score', ascending=False).head())
# print(f"Most positive message: {sentiment_trends['most_positive']['message']}") # Can be long
# print(f"Most negative message: {sentiment_trends['most_negative']['message']}") # Can be long

# 30. Word Usage Analysis (Detailed)
print("\n30. WORD USAGE ANALYSIS (Detailed)")
print("-" * 50)
word_usage = analyzer.analyze_word_usage(df)
print(f"Total words (filtered): {word_usage['total_words']}")
print(f"Word diversity (unique filtered words): {word_usage['word_diversity']}")
print(f"Average words per message (filtered): {word_usage['words_per_message']:.2f}")
if not word_usage['word_counts'].empty:
    print("Top 5 most frequent words (filtered):")
    print(word_usage['word_counts'].head())

# 31. Conversation Flow Analysis (Detailed)
print("\n31. CONVERSATION FLOW ANALYSIS (Detailed)")
print("-" * 50)
conversation_flow = analyzer.analyze_conversation_flow(df)
print(f"Total conversations identified: {conversation_flow['total_conversations']}")
print("\nConversation Statistics (first 5 conversations):")
print(conversation_flow['conversation_stats'].head())
print("\nConversation Starters (Top 5):")
print(conversation_flow['conversation_starters'].sort_values('count', ascending=False).head())
print("\nConversation Enders (Top 5):")
print(conversation_flow['conversation_enders'].sort_values('count', ascending=False).head())

# 33. Conversation Mood Shifts (Detailed)
print("\n33. CONVERSATION MOOD SHIFTS (Detailed)")
print("-" * 50)
mood_shifts = analyzer.analyze_conversation_mood_shifts(df)
if mood_shifts['mood_lifters']:
    print("Top 3 mood lifters:")
    for user, count, ratio in mood_shifts['mood_lifters'][:3]:
        print(f"  {user}: {count} lifts ({ratio*100:.2f}% lift ratio)")
if mood_shifts['mood_dampeners']:
    print("\nTop 3 mood dampeners:")
    for user, count, ratio in mood_shifts['mood_dampeners'][:3]:
        print(f"  {user}: {count} dampens ({ratio*100:.2f}% dampen ratio)")
if mood_shifts['top_shifters']:
    print("\nTop 3 overall mood shifters (by count):")
    for user, count, ratio, avg_shift in mood_shifts['top_shifters'][:3]:
        print(f"  {user}: {count} shifts ({ratio*100:.2f}% shift ratio, avg shift: {avg_shift:.2f})")


# # 32. User Activity Patterns (Detailed)  (not works as of now for future release)
# print("\n32. USER ACTIVITY PATTERNS (Detailed)")
# print("-" * 50)
# user_patterns_data = analyzer.analyze_user_activity_patterns(df)
# print("User activity patterns analyzed. Details for first 2 users:")
# for user, pattern in list(user_patterns_data['user_patterns'].items())[:2]:
#     print(f"\n  User: {user}")
#     print(f"    Total messages: {pattern['total_messages']}")
#     print(f"    Avg messages/day: {pattern['messages_per_day']:.2f}")
#     print(f"    Peak hour: {pattern['peak_hour']}")
#     print(f"    Peak day: {pattern['peak_day']}")
#     print(f"    Most responds to: {pattern['most_responds_to']} ({pattern['response_count']} times)")
#     print(f"    Most responded by: {pattern['most_responded_by']} ({pattern['responded_by_count']} times)")
#     print(f"    Longest activity streak: {pattern['longest_streak_length']} days")

# 34. Conversation Compatibility (Detailed) (not works as of now [for future release])
# print("\n34. CONVERSATION COMPATIBILITY (Detailed)")
# print("-" * 50)
# compatibility = analyzer.analyze_conversation_compatibility(df)
# if compatibility['most_compatible']:
#     print("Top 3 most compatible pairs:")
#     for user1, user2, score in compatibility['most_compatible'][:3]:
#         print(f"  {user1} & {user2}: {score:.1f}% compatibility")
# else:
#     print("Could not calculate compatibility (likely insufficient interactions).")

# # 35. Personality Analysis (Detailed) (not works as of now [for future release])
# print("\n35. PERSONALITY ANALYSIS (Detailed)")
# print("-" * 50)
# personalities = analyzer.analyze_personality(df)
# print("Personality traits for first 2 users:")
# for user, traits in list(personalities.items())[:2]:
#     print(f"\n  User: {user}")
#     print(f"    Openness: {traits['openness']}%")
#     print(f"    Conscientiousness: {traits['conscientiousness']}%")
#     print(f"    Extraversion: {traits['extraversion']}%")
#     print(f"    Agreeableness: {traits['agreeableness']}%")
#     print(f"    Neuroticism: {traits['neuroticism']}%")

# # 36. Future Activity Prediction (not works as of now [for future release])
# print("\n36. FUTURE ACTIVITY PREDICTION")
# print("-" * 50)
# prediction = analyzer.predict_future_activity(df, forecast_days=7)
# if prediction['success']:
#     print(f"Predicted messages for next 7 days: {prediction['metrics']['total_predicted_messages']}")
#     print(f"Predicted daily average: {prediction['metrics']['daily_avg_predicted']:.1f}")
#     print(f"Predicted change from historical average: {prediction['metrics']['percent_change']:.1f}%")
#     print("Predictions per day:")
#     print(prediction['predictions'])
# else:
#     print("Prediction failed:", prediction.get('error', 'Unknown error'))
#
# print("\n" + "=" * 60)
# print("Comprehensive analysis script completed successfully!")
# print("(Note: Figures are generated but not displayed by default. Uncomment '.show()' lines to view them.)")
# print("=" * 60)
