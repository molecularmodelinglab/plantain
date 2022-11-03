import pandas as pd
import seaborn as sns

def create_combined_df(csv_dict):
    dfs = []
    for (dataset, model), csv in csv_dict.items():
        df = pd.read_csv("outputs/" + csv)
        df["model"] = model
        df["dataset"] = dataset
        if dataset == "BigBind":
            df = df[df["total actives in set"] >= 1].reset_index(drop=True)
        dfs.append(df)
    return pd.concat(dfs)

def make_figures(comb_df):
    # sns.set_style("white")
    sns.set(font_scale=1.5, style='white')
    pcba_order = list(comb_df.query("dataset == 'LIT-PCBA' and model == 'BANANA'").sort_values(by="EF1%", ascending=False).target)
    bb_order = list(comb_df.query("dataset == 'BigBind' and model == 'BANANA'").sort_values(by="EF1%", ascending=False).target)
    # fig, axes = plt.subplots(1,2)
    g = sns.catplot(x="target", y="EF1%", hue="model",# col="dataset",
                    data=comb_df,
                    order=pcba_order,
                    hue_order = [ 'BANANA', 'GNINA', 'BANANA+GNINA' ],
                    # sharex = False,
                    aspect=2,
                    kind='bar')

    g.set_xticklabels(rotation=90)
    g.fig.suptitle("LIT-PCBA performance")
    out_file = "./outputs/lit_pcba.png"
    print(f"Writing figure to {out_file}")
    g.savefig(out_file)
    
def make_results_csv(comb_df):
    med_rows = []
    mean_rows = []
    for model in ["BANANA", "GNINA", "BANANA+GNINA"]:
        med_row = { "model": model }
        mean_row = { "model": model }
        for dataset in [ "BigBind", "LIT-PCBA" ]:
            for metric in [ "EF1%", "NEF1%", "auroc" ]:
                key = (dataset + metric)
                for c in "+-1%":
                    key = key.replace(c, "")
                col = comb_df.query("dataset == @dataset and model == @model")[metric]
                med_row[key] = round(col.median(), 2)
                mean_row[key] = round(col.mean(), 2)
        med_rows.append(med_row)
        mean_rows.append(mean_row)
    med_df = pd.DataFrame(med_rows)
    med_df.to_csv("./outputs/median_results.csv")

    mean_df = pd.DataFrame(mean_rows)
    mean_df.to_csv("./outputs/mean_results.csv")

    for model in ["BANANA", "BANANA+GNINA", "GNINA"]:
        for dataset in [ "BigBind", "LIT-PCBA" ]:
            dataset_df = comb_df.query("dataset == @dataset and model == @model").rename(columns = { "EF1%": "ef", "NEF1%": "nef"})
            dataset_df.to_csv(f"./outputs/{dataset}_{model}_results.csv", float_format='%.2f')

if __name__ == "__main__":
    csv_dict = {
        ("LIT-PCBA", "GNINA"): "screen_lit_pcba_test_gnina.csv",
        ("LIT-PCBA", "BANANA"): "screen_lit_pcba_test_37jstv82_v4.csv",
        ("LIT-PCBA", "BANANA+GNINA"): "screen_lit_pcba_test_combo_37jstv82_v4_gnina_0.1.csv",
        ("BigBind", "BANANA"): "screen_bigbind_test_37jstv82_v4.csv"
    }
    comb_df = create_combined_df(csv_dict)
    # make_figures(comb_df)
    make_results_csv(comb_df)

