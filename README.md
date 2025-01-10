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

# Read and preprocess your chat file
with open('chat.txt', 'r', encoding='utf-8') as file:
    chat_data = file.read()
    
# Create the DataFrame
df = preprocess(chat_data)

# Now you can use any of the analysis functions!

# Get basic stats
messages, words, media, links = analyzer.fetch_stats('Overall', df)
print(f"Total Messages: {messages}")
print(f"Total Words: {words}")
print(f"Media Messages: {media}")
print(f"Links Shared: {links}")

# Analyze user activity
fig, df_percent = analyzer.most_busy_users(df)
fig.show()  # Shows the interactive plotly visualization

# Generate word cloud
wordcloud = analyzer.create_wordcloud('Overall', df)

# Analyze sentiment
sentiments, most_positive, most_negative = analyzer.calculate_sentiment_percentage('Overall', df)
print(f"Most positive user: {most_positive}")
print(f"Most negative user: {most_negative}")

# Analyze reply patterns
user, time, msg, reply = analyzer.analyze_reply_patterns(df)
print(f"User with longest reply time: {user}")
print(f"Reply took {time:.2f} minutes")
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