import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# set the font globally
plt.rcParams.update({'font.family':'Times New Roman'})

sns.set_theme(style="whitegrid", palette="tab10")

def load_dataset(file_path):
    return pd.read_csv(file_path)

def plot_results(df_fashion, df_digit,df_oracle):
    fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(14,6), constrained_layout=True)
    # fig.tight_layout()
    axis[0, 0] = sns.lineplot(data=df_fashion[["accuracy","val_accuracy"]], ax = axis[0,0], linewidth=2.5)
    axis[0, 0].set(xlabel='Epochs')
    axis[0, 0].set_ylabel('Accuracy', weight='bold')
    axis[0, 0].set_title("Fashion-MNIST")

    axis[1, 0] = sns.lineplot(data=df_fashion[["loss","val_loss"]], ax = axis[1,0], linewidth=2.5)
    axis[1, 0].set_ylabel('Loss', weight='bold')
    axis[1, 0].set(xlabel='Epochs')

    # axis[1, 0].set_title("Fashion-MNIST")

    axis[0, 1] = sns.lineplot(data=df_digit[["accuracy","val_accuracy"]],ax = axis[0,1], linewidth=2.5)
    axis[0, 1].set_title("MNIST-Digit")
    axis[0, 1].set_xlabel('Epochs')

    axis[1, 1] = sns.lineplot(data=df_digit[["loss","val_loss"]], ax = axis[1,1], linewidth=2.5)
    # axis[1, 1].set_title("MNIST-Digit")
    axis[1, 1].set(xlabel='Epochs')

    axis[0, 2] = sns.lineplot(data=df_oracle[["accuracy","val_accuracy"]],ax = axis[0,2], linewidth=2.5)
    axis[0, 2].set_title("Oracle-MNIST")
    axis[0, 2].set(xlabel='Epochs')

    axis[1, 2] = sns.lineplot(data=df_oracle[["loss","val_loss"]], ax = axis[1,2], linewidth=2.5)
    # axis[1, 2].set_title("Oracle-MNIST")
    axis[1, 2].set(xlabel='Epochs')

    fig.text(0.19,-0.07,'(a)')
    fig.text(0.52,-0.07,'(b)')
    fig.text(0.84,-0.07,'(c)')


    plt.show()
    fig.savefig('../chart/result.pdf',bbox_inches="tight")


def run():
    print('start generating...')
    df_fashion = load_dataset('../chart/fashion-mnist-history.csv')
    df_digit = load_dataset('../chart/mnist-digit.csv')
    df_oracle = load_dataset('../chart/oracle-history.csv')


    plot_results(df_fashion, df_digit,df_oracle)

    df = pd.read_csv('../chart/comparitive.csv')
    g = sns.catplot(
    data=df, kind="bar",
    x="Database", y="Accuracy", hue="Model",
    ci="sd")
    plt.show()
    g.savefig('../chart/compare.pdf',bbox_inches="tight")
    print('file saved!')
    
if __name__ == "__main__":
    run()