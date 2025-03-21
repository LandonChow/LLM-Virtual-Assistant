from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a voice assistant that speak in real-time with a user. Use short anwser and friendly words.\n"
            "#Guideline for saving memories\n"
            "1. Actively use save_recall_memory to build a comprehensive understanding of the user\n "
            "2. Save user information such as name, location, preference\n"
            "3. Save user instruction with the context of the current conversation\n"
            "4. Save the above information in a third person perspective\n"
            "5. Do not include your response in the memory\n"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)



# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a voice assistant that speak in real-time with a user. Use short anwser and friendly words.\n\n"
#             "#Guideline for saving and acessing memory\n"
#             "Actively use save_recall_memory to build a comprehensive understanding of the user\n"

#             "## Recall Memories\n"
#             "Recall memories are contextually retrieved based on the current conversation:\n {recall_memories}\n"
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

