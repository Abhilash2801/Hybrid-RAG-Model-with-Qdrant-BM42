from groq import Groq
client = Groq(api_key='******')

class generate:
    def __init__(self):
        self.question=""
        self.context=""
        self.prompt=""
    def llm_query(self,question,context)->str:
        self.question=question
        self.context=context
        self.prompt=f"""Your task is to provide a clear, concise, and informative explanation based on the following context and query.

        Context:
        {context}

        Query: {question}

        
        Please follow these guidelines in your response:
        1. Start with a brief overview of the concept mentioned in the query.
        2. Dont mention like answer to your question or such things just the answer is enough
        3. Answer should be in 200-300 words and make it as paras if required.
        Your explanation should be informative yet accessible, suitable for someone with a basic understanding of RAG. If the query asks for information not present in the context, please state that you don't have enough information to provide a complete answer, and only respond based on the given context.
        """
        chat_title=client.chat.completions.create(messages=[{
              "role":"user",
              "content": self.prompt
          }],model="mixtral-8x7b-32768")
        return chat_title.choices[0].message.content

if __name__ == '__main__':
    search = generate()
    query = "Can you explain the objective of sustainable development?"
    context=" "
    results = search.llm_query(query,context)
    print(results)