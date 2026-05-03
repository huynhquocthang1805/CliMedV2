
from qa_engine import HFLCQAEngine

def main():
    engine = HFLCQAEngine()
    print("HFLC QA Chat v5")
    print("Gõ 'exit' để thoát.")
    while True:
        q = input("Bạn hỏi: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("Trả lời:", engine.answer(q))
        print("-" * 80)

if __name__ == "__main__":
    main()
