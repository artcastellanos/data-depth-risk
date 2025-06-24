from statistics import mean, stdev
import pandas as pd
from AUCroutine import *

if __name__ == '__main__':
    method = {'name':'ParallelSVMDepth','surname':'$SVM\mathcal{D}$','func':ParallelSVMDepthroutine}#{'name':'ParallelLogDepth','surname':'$LR\mathcal{D}$','func':ParallelLogDepthroutine}
    dataname = 'mnist_784'#'Fashion-MNIST'#
    lamb = 1
    max_iter = 1000
    numX = 100#1000#None#
    numZ_normal = None
    numZ_anormal = None

    seed_list = [0]
    results = []

    row_name = method['surname']
    if numX is not None:
        row_name = row_name + ' ($n='+str(numX)+'$)'
        seed_list = range(10)

    # just for initialisation of params
    y_class = '1'
    seed = 42

    params = {'seed':seed,
        'y_class':y_class,
        'lamb':lamb ,
        'max_iter':max_iter,
        'numX':numX,
        'numZ_normal':numZ_normal,
        'numZ_anormal':numZ_anormal}

    y_list = [str(x) for x in range(10)]
    col = y_list + ["mean"]
    df = pd.DataFrame(index = [row_name], columns=col)

    for y_class in y_list:
            params['y_class'] = y_class
            print("~~~~~~~~~~~~~~~~~~~~~")
            print("class", y_class)
            class_results = []
            for seed in seed_list:
                params['seed'] = seed
                paramsname = paramdict_toString(params)
                save_path = 'results/AUC/' + method['name'] + dataname + '_' + paramsname
                score = joblib.load(save_path)
                print("seed",seed)
                print("depth",round(score,2))
                results.append(score)
                class_results.append(score)
            class_mean = round(mean(class_results),2)
            print("class_mean",class_mean)
            print()
            df[y_class] = class_mean
    df["mean"] = round(mean(results),2)
    print(df)
    with open("results/AUC/" + dataname + method['name'] + '_numX_'+str(numX) + ".tex", "w") as fh:
        df.style.format(precision=2).to_latex(fh)