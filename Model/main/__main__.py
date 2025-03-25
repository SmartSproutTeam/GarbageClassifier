import time
import pandas as pd
from main.model import build_model, test_model, train_model
from main.preprocess import make_subsets

if __name__ == "__main__":
    start_time = time.time()

    X_train, y_train, X_test, y_test, label_names = make_subsets()

    print("Finished loading")

    model = build_model(len(label_names))
    history = train_model(model, X_train, y_train)

    print("Finished training")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal training time: {elapsed_time:.2f} seconds")

    test_model(model, X_test, y_test)

    model.save("best_model.keras")
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("training_history.csv", index=False)
