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

print("WhatsApp Chat Analysis")
print("=" * 50)

# 1. Basic Statistics
print("\n1. BASIC STATISTICS")
print("-" * 50)
messages, words, media, links = analyzer.fetch_stats('Overall', df)
print(f"Total Messages: {messages}")
print(f"Total Words: {words}")
print(f"Media Messages: {media}")
print(f"Links Shared: {links}")

# # 2. User Activity Analysis
# print("\n2. USER ACTIVITY ANALYSIS")
# print("-" * 50)
# fig, df_percent = analyzer.most_busy_users(df)
# print("Most Active Users (% of total messages):")
# for index, row in df_percent.head().iterrows():
#     print(f"{row['name']}: {row['percent']}%")

# # 3. Word Cloud Generation
# print("\n3. WORD CLOUD GENERATION")
# print("-" * 50)
# wordcloud = analyzer.create_wordcloud('Overall', df)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.title("Word Cloud")
# plt.savefig('wordcloud.png')
# print("Word cloud generated and saved as 'wordcloud.png'")

# 4. Most Common Words
print("\n4. MOST COMMON WORDS")
print("-" * 50)
common_words_fig = analyzer.most_common_words('Overall', df)
print("Most common words analysis completed")

# # 5. Emoji Analysis
# print("\n5. EMOJI ANALYSIS")
# print("-" * 50)
# emoji_fig = analyzer.emoji_analysis('Overall', df)
# print("Emoji analysis completed")

# 6. Timeline Analysis
print("\n6. TIMELINE ANALYSIS")
print("-" * 50)
monthly_timeline = analyzer.monthly_timeline('Overall', df)
daily_timeline = analyzer.daily_timeline('Overall', df)
print("Timeline analysis completed")

# 7. Activity Heatmap
print("\n7. ACTIVITY HEATMAP")
print("-" * 50)
activity_heatmap = analyzer.activity_heatmap('Overall', df)
print("Activity patterns by day and hour:")
print(activity_heatmap)

# 8. Sentiment Analysis
print("\n8. SENTIMENT ANALYSIS")
print("-" * 50)
sentiments, most_positive, most_negative = analyzer.calculate_sentiment_percentage('Overall', df)
print(f"Most positive user: {most_positive}")
print(f"Most negative user: {most_negative}")

# 9. Reply Pattern Analysis
print("\n9. REPLY PATTERN ANALYSIS")
print("-" * 50)
user, time, msg, reply = analyzer.analyze_reply_patterns(df)
print(f"User with longest reply time: {user}")
print(f"Reply took {time:.2f} minutes")
print(f"Original message: {reply}")
print(f"Reply message: {msg}")

# # 10. Message Types Analysis
# print("\n10. MESSAGE TYPES ANALYSIS")
# print("-" * 50)
# message_types = analyzer.analyze_message_types('Overall', df)
# print("Message type distribution:")
# for msg_type, count in message_types.items():
#     print(f"{msg_type}: {count}")

# 11. Conversation Pattern Analysis
# print("\n11. CONVERSATION PATTERN ANALYSIS")
# print("-" * 50)
# conversation_patterns = analyzer.analyze_conversation_patterns(df)
# print(f"Total conversations: {conversation_patterns['total_conversations']}")
# print(f"Average conversation length: {conversation_patterns['avg_conversation_length']:.2f} messages")
# print(f"Average conversation duration: {conversation_patterns['avg_conversation_duration_mins']:.2f} minutes")

# # 12. User Interaction Analysis
# print("\n12. USER INTERACTION ANALYSIS")
# print("-" * 50)
# G, interactions = analyzer.analyze_user_interactions(df)
# print(f"Number of users: {len(G.nodes())}")
# print(f"Number of interactions: {len(G.edges())}")
#
# print("doneeee")

# 13. Time Pattern Analysis
print("\n13. TIME PATTERN ANALYSIS")
print("-" * 50)
time_patterns = analyzer.analyze_time_patterns(df)
print("Hourly activity analysis completed")
print("Daily activity analysis completed")
print("Monthly activity analysis completed")

# 14. Message Length Analysis
print("\n14. MESSAGE LENGTH ANALYSIS")
print("-" * 50)
message_length = analyzer.analyze_message_length(df)
print(f"Average message length: {message_length['avg_length']:.2f} words")
print(f"Maximum message length: {message_length['max_length']} words")
print(f"Minimum message length: {message_length['min_length']} words")
print(f"Longest message by: {message_length['longest_message']['user']}")

# # 15. Response Time Analysis date ambigous
# print("\n15. RESPONSE TIME ANALYSIS")
# print("-" * 50)
# response_times = analyzer.analyze_response_times(df)
# print(f"Average response time: {response_times['avg_response_time']:.2f} minutes")
# print(f"Median response time: {response_times['median_response_time']:.2f} minutes")

# 16. Topic Modeling
print("\n16. TOPIC MODELING")
print("-" * 50)
topics = analyzer.analyze_topic_modeling(df, num_topics=3, num_words=5)
if topics['success']:
    print("Topics extracted:")
    for topic in topics['topics']:
        print(f"Topic {topic['topic_id']}: {', '.join(topic['words'])}")
else:
    print("Topic modeling failed:", topics.get('error', 'Unknown error'))

# # 17. Emoji Usage Analysis
# print("\n17. EMOJI USAGE ANALYSIS")
# print("-" * 50)
# emoji_usage = analyzer.analyze_emoji_usage(df)
# print(f"Total emojis used: {emoji_usage['total_emojis']}")
# print(f"Emoji diversity: {emoji_usage['emoji_diversity']} unique emojis")
# print(f"Emoji density: {emoji_usage['emoji_density']:.2f} emojis per message")
# if len(emoji_usage['emoji_counts']) > 0:
#     print("Top 5 emojis:")
#     for _, row in emoji_usage['emoji_counts'].head(5).iterrows():
#         print(f"{row['emoji']}: {row['count']}")

# 18. Sentiment Trend Analysis
print("\n18. SENTIMENT TREND ANALYSIS")
print("-" * 50)
sentiment_trends = analyzer.analyze_sentiment_trends(df)
print("Sentiment distribution:")
for sentiment, count in sentiment_trends['sentiment_counts'].items():
    print(f"{sentiment}: {count}")
print(f"Average sentiment score: {sentiment_trends['avg_sentiment']:.2f}")
print(f"Most positive message: {sentiment_trends['most_positive']['message']}")
print(f"Most negative message: {sentiment_trends['most_negative']['message']}")

# # 19. Word Usage Analysis date ambigous
# print("\n19. WORD USAGE ANALYSIS")
# print("-" * 50)
# word_usage = analyzer.analyze_word_usage(df)
# print(f"Total words: {word_usage['total_words']}")
# print(f"Word diversity: {word_usage['word_diversity']} unique words")
# print(f"Words per message: {word_usage['words_per_message']:.2f}")
# if len(word_usage['word_counts']) > 0:
#     print("Top 5 words:")
#     for _, row in word_usage['word_counts'].head(5).iterrows():
#         print(f"{row['word']}: {row['count']}")

# # 20. Conversation Flow Analysis date ambigous
# print("\n20. CONVERSATION FLOW ANALYSIS")
# print("-" * 50)
# conversation_flow = analyzer.analyze_conversation_flow(df)
# print(f"Total conversations: {conversation_flow['total_conversations']}")
# print("Conversation starters:")
# for _, row in conversation_flow['conversation_starters'].head(3).iterrows():
#     print(f"{row['user']}: {row['count']} conversations started")

# # 21. User Activity Patterns
# print("\n21. USER ACTIVITY PATTERNS")
# print("-" * 50)
# user_patterns = analyzer.analyze_user_activity_patterns(df)
# print("User activity patterns analyzed")
# for user, pattern in list(user_patterns['user_patterns'].items())[:2]:  # Show first 2 users
#     print(f"\nUser: {user}")
#     print(f"Total messages: {pattern['total_messages']}")
#     print(f"Messages per day: {pattern['messages_per_day']:.2f}")
#     print(f"Peak hour: {pattern['peak_hour']}")
#     print(f"Peak day: {pattern['peak_day']}")
#
# 22. Conversation Mood Shifts
print("\n22. CONVERSATION MOOD SHIFTS")
print("-" * 50)
mood_shifts = analyzer.analyze_conversation_mood_shifts(df)
if mood_shifts['mood_lifters']:
    print("Top mood lifters:")
    for user, count, ratio in mood_shifts['mood_lifters'][:3]:
        print(f"{user}: {count} mood lifts ({ratio*100:.2f}%)")

# 23. Conversation Compatibility
print("\n23. CONVERSATION COMPATIBILITY")
print("-" * 50)
compatibility = analyzer.analyze_conversation_compatibility(df)
if compatibility['most_compatible']:
    print("Most compatible pairs:")
    for user1, user2, score in compatibility['most_compatible'][:3]:
        print(f"{user1} & {user2}: {score:.2f}% compatibility")

# 24. Personality Analysis
print("\n24. PERSONALITY ANALYSIS")
print("-" * 50)
personalities = analyzer.analyze_personality(df)
for user, traits in list(personalities.items())[:2]:  # Show first 2 users
    print(f"\nUser: {user}")
    print(f"Openness: {traits['openness']}%")
    print(f"Conscientiousness: {traits['conscientiousness']}%")
    print(f"Extraversion: {traits['extraversion']}%")
    print(f"Agreeableness: {traits['agreeableness']}%")
    print(f"Neuroticism: {traits['neuroticism']}%")

# 25. Future Activity Prediction
print("\n25. FUTURE ACTIVITY PREDICTION")
print("-" * 50)
prediction = analyzer.predict_future_activity(df, forecast_days=7)
if prediction['success']:
    print(f"Predicted messages for next 7 days: {prediction['metrics']['total_predicted_messages']}")
    print(f"Daily average prediction: {prediction['metrics']['daily_avg_predicted']}")
    print(f"Predicted change from historical average: {prediction['metrics']['percent_change']}%")
else:
    print("Prediction failed:", prediction.get('error', 'Unknown error'))

print("\nAnalysis completed successfully!")
