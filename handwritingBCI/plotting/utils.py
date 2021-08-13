import matplotlib.pyplot as plt


def plot_electrode_data(data, label, index=None, title_size=25,
                        figsize=(15, 15), deproc_label=None):
    if index is not None:
        data = data[index]
        label = label[index]
    if deproc_label:
        label = deproc_label(label)
    plt.figure(figsize=figsize)
    plt.imshow(data.T)
    plt.title(f"character: {label}",size=title_size)
    plt.xlabel("Time steps", size= title_size)
    plt.ylabel("Electrodes", size=title_size)
    plt.show()
