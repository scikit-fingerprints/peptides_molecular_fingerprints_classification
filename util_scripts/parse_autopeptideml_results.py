import pandas as pd

"""
AutoPeptideML results are originally a weirdly shaped CSV. Here, we transform
it into nicer tabular form and calculate average performance of each method.
"""

if __name__ == "__main__":
    df = pd.read_csv("data/autopeptideml_results.csv")
    models = [col for col in df.columns if col not in ["Dataset", "Metric"]]

    model_values = []
    for model_name in models:
        print(model_name)
        model_dict = {"model": model_name}
        for metric_name in ["ACC", "MCC", "AUROC", "F1"]:
            values = df.loc[df["Metric"] == metric_name, model_name]
            values = values.apply(lambda x: float(x.split(" ± ")[0]))
            mean = values.mean().round(3)
            std = values.std().round(3)
            print(f"\t{metric_name}: {mean:.3f} +- {std:.3f}")
            model_dict[metric_name] = mean

        model_values.append(model_dict)

    df = pd.DataFrame(model_values)
    df = df.sort_values(by="MCC", ascending=False)
    print(df)
