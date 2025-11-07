"""DSPy signatures for Neibot service."""

import dspy


class MessageSignature(dspy.Signature):
    """Process a single message with text and images."""

    content: str | dspy.Image = dspy.InputField(desc="Message to process")


class ConversationSignature(dspy.Signature):
    """Answer user questions based on conversation context and system instructions."""

    system_prompt = dspy.InputField(desc="System instructions and personality")
    context: list[MessageSignature] = dspy.InputField(
        desc="Previous conversation history"
    )
    question = dspy.InputField(desc="Current user question or message")
    answer = dspy.OutputField(desc="Response to the user")
