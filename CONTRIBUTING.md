# Contributing to Sus Message Bot

Thank you for your interest in contributing! Here's how you can help:

## Adding New Examples

The most impactful contribution is adding real-world scam/spam examples to improve the bot's accuracy.

1. Add your examples to `susmessagebot/seeds.py` in the following format:

```python
("your example message here", "BAN"),  # or "SAFE"
```

2. Ensure examples are real-world and not fabricated
3. Include both BAN and SAFE examples where possible

## Improving the Prompt

If you find cases where the bot is incorrectly classifying messages:

1. Open an issue describing the false positive/negative
2. Include the message that was misclassified
3. Suggest how the system prompt could be improved

## Code Contributions

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Ensure code follows existing structure and includes docstrings
5. Open a pull request with a clear description of changes

## Guidelines

- Never commit API keys or tokens
- Test your changes locally before submitting a PR
- Keep code changes minimal and focused

<!-- Cursor multi-file change demo: contributing guide -->
