# WhatsApp Reality

A comprehensive Python library for analyzing WhatsApp chat exports. This library provides tools for preprocessing WhatsApp chat data and performing various analyses including sentiment analysis, user activity patterns, conversation patterns, and more.

## Installation

```bash
pip install whatsapp-reality
```

## Features

- Chat preprocessing for both Android and iOS WhatsApp exports
- Basic statistics (message counts, word counts, media counts)
- User activity analysis
- Word clouds and common words analysis
- Sentiment analysis
- Emoji analysis
- Timeline analysis (daily, monthly)
- Reply time analysis
- Conversation pattern analysis

## Quick Start

### 1. Export your WhatsApp chat

1. Open WhatsApp
2. Go to the chat you want to analyze
3. Click on the three dots (⋮) > More > Export chat
4. Choose "Without Media"
5. Save the exported text file

### 2. Analyze your chat

```python
from whatsapp_reality import preprocess, analyzer
import pandas as pd # Assuming pandas is needed for DataFrame display

# Read and preprocess your chat file
with open('chat.txt', 'r', encoding='utf-8') as file:
    chat_data = file.read()

# Create the DataFrame
df = preprocess(chat_data)

# Assume 'selected_user' is defined, e.g., 'Overall' or a specific user name
selected_user = 'Overall' # Example: Replace with actual user or 'Overall'

# Now you can use any of the analysis functions!

# --- Basic Stats ---
messages, words, media, links = analyzer.fetch_stats(selected_user, df)
print(f"Total Messages: {messages}")
print(f"Total Words: {words}")
print(f"Media Messages: {media}")
print(f"Links Shared: {links}")

# --- User Activity ---
busy_users_fig, df_percent = analyzer.most_busy_users(df)
# busy_users_fig.show() # Shows the interactive plotly visualization

# --- Time-based Analysis ---
monthly_timeline_df = analyzer.monthly_timeline(selected_user, df)
print("\nMonthly Timeline:")
print(monthly_timeline_df)

daily_timeline_df = analyzer.daily_timeline(selected_user, df)
print("\nDaily Timeline:")
print(daily_timeline_df)

weekly_activity_fig = analyzer.week_activity_map(selected_user, df)
# weekly_activity_fig.show() # Uncomment to display plot

monthly_activity_fig = analyzer.month_activity_map(selected_user, df)
# monthly_activity_fig.show() # Uncomment to display plot

busiest_hours_fig = analyzer.busiest_hours_analysis(selected_user, df)
# busiest_hours_fig.show() # Uncomment to display plot

# --- Content Analysis ---
wordcloud_fig = analyzer.create_plotly_wordcloud(selected_user, df)
# wordcloud_fig.show() # Uncomment to display plot

common_words_df = analyzer.most_common_words(selected_user, df)
print("\nMost Common Words:")
print(common_words_df)

emoji_df = analyzer.emoji_helper(selected_user, df)
print("\nEmoji Usage:")
print(emoji_df)

# --- Sentiment Analysis ---
positive_fig, negative_fig = analyzer.analyze_and_plot_sentiment(selected_user, df)
# positive_fig.show() # Uncomment to display plot
# negative_fig.show() # Uncomment to display plot

sentiment_percentages, most_positive, least_positive = analyzer.calculate_sentiment_percentage(selected_user, df)
print("\nSentiment Percentages:")
print(sentiment_percentages)
print("Most Positive Message Snippet:", most_positive)
print("Least Positive Message Snippet:", least_positive)

monthly_sentiment_fig = analyzer.calculate_monthly_sentiment_trend(selected_user, df)
# monthly_sentiment_fig.show() # Uncomment to display plot

# --- User Interaction Analysis ---
msg_count_fig, who_do_most_messages = analyzer.message_count_aggregated_graph(df)
# msg_count_fig.show() # Uncomment to display plot
print("\nUser with most messages:", who_do_most_messages)

starter_fig, who_starts_convo = analyzer.conversation_starter_graph(df)
# starter_fig.show() # Uncomment to display plot
print("User who starts most conversations:", who_starts_convo)

convo_size_fig = analyzer.conversation_size_aggregated_graph(df)
# convo_size_fig.show() # Uncomment to display plot

late_reply_fig, avg_late_reply_times, overall_avg = analyzer.calculate_average_late_reply_time(df)
# late_reply_fig.show() # Uncomment to display plot
print("\nAverage late reply times per user:")
print(avg_late_reply_times)
print("Overall average late reply time:", overall_avg)

# Example from original Quick Start (if different from above)
# analyze_reply_patterns might be deprecated or replaced by calculate_average_late_reply_time
# user, time, msg, reply = analyzer.analyze_reply_patterns(df)
# print(f"\nUser with longest reply time: {user}")
# print(f"Reply took {time:.2f} minutes")

```

## Supported Chat Formats

The library supports WhatsApp chat exports from both Android and iOS devices in the following formats:

### Android Format
```
DD/MM/YY, HH:mm - Username: Message
```

### iOS Format
```
[DD/MM/YY, HH:mm:ss] Username: Message
```

Both 12-hour and 24-hour time formats are supported.

## Documentation

For detailed documentation and examples, visit our [documentation page](https://github.com/Abdul1028/whatsapp-reality).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- Abdul
- [GitHub Profile](https://github.com/Abdul1028)

## Acknowledgments

Special thanks to all contributors and users of this library. 