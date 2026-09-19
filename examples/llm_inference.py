from gatekg.api.llm import LLM

if __name__ == "__main__":
    llm = LLM()
    print(llm.predict("What is the capital of Romania?"))
