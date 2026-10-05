from router.core import LLMRouter


def main():
    router = LLMRouter()

    print("LLM Router")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        result = router.route(query)

        print(f"\nAssistant: {result.response}")
        print(f"Model: {result.model_used}")
        print(f"Cache hit: {result.cache_hit}")
        print(f"Complexity: {result.complexity_score}")
        print(f"Escalated: {result.escalated}")
        print(f"Cost: ${result.cost_usd:.6f}")
        print(f"Latency: {result.latency_s:.2f}s\n")


if __name__ == "__main__":
    main()