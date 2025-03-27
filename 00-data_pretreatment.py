import argparse
import pickle
from utilities import (create_rawDataFrames,
                       load_rawDataFrames,
                       loadXY,
                       )

def create_and_preprocess_data():
    parser = argparse.ArgumentParser(description="Create the pre-treated dataframes offering a two steps process.")
    parser.add_argument("-s", "--srcPath",
                        type=str,
                        default="src/",
                        help='The path to the src csv files.'
                        )
    parser.add_argument("-raw", "--createraw",
                        action="store_true",
                        help='if this flag is set, the script only performs the raw dataframes creation.'
                        )
    parser.add_argument("-pp", "--preprocess",
                        action="store_true",
                        help='if this flag is set, the script only performs the data preprocessing.'
                        )
    args = parser.parse_args()

    if not args.preprocess:
        dfs = create_rawDataFrames(args.srcPath)
        if not args.createraw:
            print("(by default) Creating the raw dataFrames and preprocessing the dataFrames.")
        if args.createraw:
            print("Creating the raw dataFrames.")
            return

    if args.preprocess:
        dfs = load_rawDataFrames()
        print("Preprocessing the dataFrames.")
    
    X_tabular_train, X_time_train, y_target_train = loadXY(dfs, "train", use_prev_year=True)
    print("train shape", X_time_train.shape)
    X_tabular_validation, X_time_valid, y_target_valid, valid_fips = loadXY(dfs, "validation", return_fips=True, use_prev_year=True)
    print("validation shape", X_time_valid.shape)
    X_tabular_test, X_time_test, y_target_test, test_fips = loadXY(dfs, "test", return_fips=True, use_prev_year=True)
    print("test shape", X_time_test.shape)

    data = {}
    data["X_tabular_train"] = X_tabular_train
    data["X_time_train"] = X_time_train
    data["y_target_train"] = y_target_train
    data["X_tabular_validation"] = X_tabular_validation
    data["X_time_validation"] = X_time_valid
    data["y_target_validation"] = y_target_valid
    data["valid_fips"] = valid_fips
    data["X_tabular_test"] = X_tabular_test
    data["X_time_test"] = X_time_test
    data["y_target_test"] = y_target_test
    data["test_fips"] = test_fips

    with open("data/data.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    create_and_preprocess_data()