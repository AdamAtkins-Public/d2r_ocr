import os
import matplotlib.pyplot as plt
import pandas as pd
import text_detection
import math

def process_session(session_dir_path,text_detector):
    """
        generate bar graph, box plot, and table from session exp data
    """

    queries = []
    with os.scandir(session_dir_path) as entries:
        session_name = session_dir_path.rsplit("\\")[-1]
        for entry in entries:
            query = []
            if not entry.name.startswith('.') and entry.name.endswith('.png'):
                query.append(entry.path)
                query.append(entry.name)
                try:
                    query.append(entry.stat().st_ctime_ns)
                except AttributeError:
                    #[TODO] log
                    pass
            queries.append(query)

    #queries = [[image_path,image_name,file_creat_time]]
    image_detection_box_pairs = []
    for query in queries:
        image_detection_box_pairs.append(text_detector.detect_image(query[0]))

    image_exp_value_pairs = text_detection.tesseract_experience_value(image_detection_box_pairs)

    # results = [[[image_path,image_name,file_creat_time],[image,exp_value]]]
    results = list(zip(queries,image_exp_value_pairs))

    #data = [image_name,delta_exp,delta_time]
    dataset = []
    for index in range(1,len(results)):
        data = []
        data.append(results[index][0][1])
        data.append(results[index][1][1] - results[index-1][1][1])
###
### Approximation
###
        #convert ns to min, truncated to 2 figures beyond the decimal
        data.append(float(format((results[index][0][2] - results[index-1][0][2])/(6*math.pow(10,10)), '.2f')))
        dataset.append(data)

    #print(dataset)

    #bargraph: experience gained per segment
    fig, ax = plt.subplots()

    segments = [data[0] for data in dataset]
    values = [data[1] for data in dataset]

    plt.xticks(rotation=90)
    ax.bar(segments,values)
    ax.set_ylabel('Experience gained')
    ax.set_title('Experienced gained per segment')

    fig.savefig(session_dir_path + "\\bargraph.png")
    plt.close(fig)

    #boxplot: experience gained per session
    fig, ax = plt.subplots()

    values = [data[1] for data in dataset]
    labels = [session_name]

    ax.boxplot(values)#,tick_labels=labels) ##feature broken v3.7.5 11/30/2024
    ax.set_ylabel('Experience gained')
    ax.set_title('Experienced gained per session')

    fig.savefig(session_dir_path + "\\boxplot.png")
    plt.close(fig)

    #table: row: segment col: delta exp, delta time min, average exp/min
    delta_exp = [data[1] for data in dataset]
    delta_time = [data[2] for data in dataset]
    exp_per_minute = [int(data[1]/data[2]) for data in dataset]
    row_labels = [data[0] for data in dataset]
    col_labels = ['Experience','Duration (min)','Experience/min']

    segments = pd.DataFrame({
                col_labels[0] : delta_exp,
                col_labels[1] : delta_time,
                col_labels[2] : exp_per_minute
                },index = row_labels)

    averages = pd.DataFrame({
                "Average Experience": int(sum(delta_exp)/len(delta_exp)),
                "Average Duration": float(format(sum(delta_time)/len(delta_time), '.2f'))
                },index=[0])

    with open(session_dir_path + "\\table.txt",mode='w') as f:
        print(segments, file=f)
        print(averages, file=f)

if __name__ == '__main__':
    #test
    pass

